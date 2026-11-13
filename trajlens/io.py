"""Loading joint trajectories from CSV, JSON, JSON lines and (optionally) rosbag2."""
import csv
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

TIME_NAMES = ["t", "time", "stamp", "timestamp", "sec", "time_s", "time_from_start"]


class TrajError(ValueError):
    """A problem with the input that the user can fix; the CLI prints it and exits 2."""


@dataclass
class Trajectory:
    t: np.ndarray                       # (N,) seconds, strictly increasing
    q: np.ndarray                       # (N, J) joint positions, rad
    names: list                         # J joint names
    source: str = ""
    notes: list = field(default_factory=list)   # things the loader fixed, shown to the user


def _finish(t, q, names, source, t0=None, t1=None, notes=None):
    notes = list(notes or [])
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or len(t) != len(q) or q.shape[1] < 1:
        raise TrajError(f"{source}: expected one time per row and at least one joint column")
    good = np.isfinite(t) & np.all(np.isfinite(q), axis=1)
    if not good.all():
        notes.append(f"dropped {int((~good).sum())} row(s) with NaN or inf")
        t, q = t[good], q[good]
    order = np.argsort(t, kind="stable")
    if not np.all(order == np.arange(len(order))):
        notes.append("time was not increasing; rows sorted by time")
        t, q = t[order], q[order]
    keep = np.concatenate([[True], np.diff(t) > 0]) if len(t) else np.array([], dtype=bool)
    if not keep.all():
        notes.append(f"dropped {int((~keep).sum())} row(s) with a repeated timestamp")
        t, q = t[keep], q[keep]
    if t0 is not None or t1 is not None:
        m = np.ones(len(t), dtype=bool)
        if t0 is not None:
            m &= t >= t0
        if t1 is not None:
            m &= t <= t1
        t, q = t[m], q[m]
    if len(t) < 3:
        raise TrajError(f"{source}: need at least 3 samples, found {len(t)}")
    return Trajectory(t, q, list(names), source, notes)


# ---------------------------------------------------------------- CSV

def _is_time(name):
    return name.strip().lower() in TIME_NAMES


def load_csv(path, cols=None, joints=None, rate=None, **kw):
    cols = cols or {}
    try:
        with open(path, newline="") as f:
            rows = [r for r in csv.reader(f) if r]
    except FileNotFoundError:
        raise TrajError(f"{path}: file not found") from None
    except IsADirectoryError:
        raise TrajError(f"{path}: is a directory, expected a file") from None
    except PermissionError:
        raise TrajError(f"{path}: permission denied") from None
    except (UnicodeDecodeError, csv.Error):
        raise TrajError(f"{path}: not a text CSV file") from None
    if len(rows) < 2:
        raise TrajError(f"{path}: no data rows")
    header = [h.strip() for h in rows[0]]
    tcol = None
    if "t" in cols:
        if cols["t"] not in header:
            raise TrajError(f"{path}: time column '{cols['t']}' not found; headers: {header}")
        tcol = header.index(cols["t"])
    else:
        tcol = next((i for i, h in enumerate(header) if _is_time(h)), None)
    if tcol is None and rate is None:
        raise TrajError(f"{path}: no time column found; headers: {header} (use --cols t=<header> or --rate HZ)")
    if joints:
        missing = [j for j in joints if j not in header]
        if missing:
            raise TrajError(f"{path}: joint column(s) {missing} not found; headers: {header}")
        jidx = [header.index(j) for j in joints]
    else:
        jidx = [i for i in range(len(header)) if i != tcol]
    if not jidx:
        raise TrajError(f"{path}: no joint columns found")
    try:
        data = np.array([[float(r[i]) for i in ([tcol] if tcol is not None else []) + jidx] for r in rows[1:]], dtype=float)
    except (ValueError, IndexError) as e:
        raise TrajError(f"{path}: a time or joint column is not numeric ({e}); pick columns with --joints") from None
    if tcol is None:
        t = np.arange(len(data)) / float(rate)
        q = data
    else:
        t, q = data[:, 0], data[:, 1:]
    return _finish(t, q, [header[i] for i in jidx], path, notes=[] if tcol is not None else [f"no time column; assumed {rate} Hz"], **kw)


# ---------------------------------------------------------------- JSON

def _stamp(v):
    """A time as seconds from a number or a ROS-style {sec, nanosec} dict."""
    if isinstance(v, dict):
        return float(v.get("sec", v.get("secs", 0))) + 1e-9 * float(v.get("nanosec", v.get("nsec", 0)))
    return float(v)


def _from_points(points, names, path, **kw):
    t, q, nm = [], [], list(names or [])
    for i, p in enumerate(points):
        pos = p.get("positions", p.get("position", p.get("pos")))
        ts = p.get("time_from_start", p.get("t", p.get("time")))
        if pos is None or ts is None:
            raise TrajError(f"{path}: point {i} needs 'positions' and 'time_from_start' (or 't')")
        if isinstance(pos, dict):
            if not nm:
                nm = list(pos)
            pos = [pos[n] for n in nm]
        t.append(_stamp(ts))
        q.append(pos)
    if not nm:
        nm = [f"joint{i + 1}" for i in range(len(q[0]))]
    return _finish(t, q, nm, path, **kw)


def load_json(path, **kw):
    try:
        text = open(path).read()
    except FileNotFoundError:
        raise TrajError(f"{path}: file not found") from None
    except (IsADirectoryError, PermissionError, UnicodeDecodeError) as e:
        raise TrajError(f"{path}: cannot read ({type(e).__name__})") from None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # JSON lines: the armdoc recorder writes {"type": "joints", "t": ..., "pos": {...}} lines
        pts = []
        for ln, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                raise TrajError(f"{path}: line {ln} is not JSON") from None
            if d.get("type", "joints") == "joints" and ("pos" in d or "positions" in d):
                pts.append(d)
        if not pts:
            raise TrajError(f"{path}: no joint samples found (expected JSON, or JSON lines with 'pos' dicts)")
        return _from_points(pts, None, path, **kw)
    if isinstance(obj, list):
        return _from_points(obj, None, path, **kw)
    if "points" in obj:
        return _from_points(obj["points"], obj.get("joint_names"), path, **kw)
    if "t" in obj and "q" in obj:
        q = obj["q"]
        if isinstance(q, dict):
            names = list(q)
            return _finish(obj["t"], np.array([q[n] for n in names]).T, names, path, **kw)
        return _finish(obj["t"], q, obj.get("names") or [f"joint{i + 1}" for i in range(len(q[0]))], path, **kw)
    raise TrajError(f"{path}: JSON needs 'points' (joint_trajectory), or 't' and 'q'")


# ---------------------------------------------------------------- rosbag2 (optional)

def bag_available():
    try:
        import rosbag2_py  # noqa: F401
        from rclpy.serialization import deserialize_message  # noqa: F401
        from rosidl_runtime_py.utilities import get_message  # noqa: F401
        return True
    except Exception:
        return False


def load_bag(path, topic=None, **kw):
    if not bag_available():
        raise TrajError(f"{path}: reading rosbag2 needs rosbag2_py (source a ROS 2 install); "
                        "or export the topic to CSV, e.g. with ros2 topic echo --csv")
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    storage = "mcap" if path.endswith(".mcap") else "sqlite3"
    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=storage), rosbag2_py.ConverterOptions("", ""))
    except RuntimeError as e:
        raise TrajError(f"{path}: cannot open the bag ({e})") from None
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    ok = ("sensor_msgs/msg/JointState", "trajectory_msgs/msg/JointTrajectory")
    if topic is None:
        cands = [n for n, ty in types.items() if ty in ok]
        topic = "/joint_states" if "/joint_states" in cands else (cands[0] if cands else None)
    if topic is None or topic not in types:
        raise TrajError(f"{path}: no JointState or JointTrajectory topic; topics: {sorted(types)} (use --topic)")
    msg_type = get_message(types[topic])
    pts, names, t0 = [], None, None
    while reader.has_next():
        tp, data, stamp = reader.read_next()
        if tp != topic:
            continue
        m = deserialize_message(data, msg_type)
        if types[topic].endswith("JointState"):
            names = list(m.name)
            hdr = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
            pts.append({"t": hdr, "bag_t": stamp * 1e-9, "positions": list(m.position)})
        else:
            names = list(m.joint_names)
            for p in m.points:
                pts.append({"t": p.time_from_start.sec + 1e-9 * p.time_from_start.nanosec, "positions": list(p.positions)})
    if not pts:
        raise TrajError(f"{path}: topic {topic} has no messages")
    if any("bag_t" in p for p in pts) and not all(p["t"] > 0 for p in pts):
        for p in pts:                                    # header stamps missing: use the time the bag recorded the message
            p["t"] = p["bag_t"]
    return _from_points(pts, names, path, **kw)


# ---------------------------------------------------------------- entry point

def load(path, fmt=None, cols=None, joints=None, rate=None, topic=None, t0=None, t1=None):
    """Load a trajectory. fmt is 'csv', 'json' or 'bag'; guessed from the extension when omitted."""
    kw = dict(t0=t0, t1=t1)
    if fmt is None:
        low = path.lower()
        if os.path.isdir(path) or low.endswith((".db3", ".mcap")):
            fmt = "bag"
        elif low.endswith((".json", ".jsonl", ".ndjson")):
            fmt = "json"
        else:
            fmt = "csv"
    if fmt == "csv":
        tr = load_csv(path, cols=cols, joints=joints, rate=rate, **kw)
    elif fmt == "json":
        tr = load_json(path, **kw)
    elif fmt == "bag":
        tr = load_bag(path, topic=topic, **kw)
    else:
        raise TrajError(f"unknown format '{fmt}', expected csv, json or bag")
    if joints and fmt != "csv":
        idx = [tr.names.index(j) for j in joints if j in tr.names]
        if len(idx) != len(joints):
            raise TrajError(f"{path}: joints {[j for j in joints if j not in tr.names]} not in {tr.names}")
        tr = Trajectory(tr.t, tr.q[:, idx], [tr.names[i] for i in idx], tr.source, tr.notes)
    for n in tr.notes:
        print(f"trajlens: {path}: {n}", file=sys.stderr)
    return tr
