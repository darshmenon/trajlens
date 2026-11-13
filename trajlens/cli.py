"""Command line: trajlens analyze | compare | track."""
import argparse
import json
import sys

from trajlens import __version__
from trajlens.findings import Config, trajectory_findings, worst_level
from trajlens.io import TrajError, load
from trajlens.metrics import analyze, parse_urdf_limits


def _common(p):
    p.add_argument("--joints", help="comma separated joint names to keep")
    p.add_argument("--cols", help="comma separated CSV columns to use as joints")
    p.add_argument("--rate", type=float, help="sample rate in Hz when the CSV has no time column")
    p.add_argument("--topic", help="rosbag2 topic (joint_states or a trajectory topic)")
    p.add_argument("--t0", type=float, help="ignore samples before this time")
    p.add_argument("--t1", type=float, help="ignore samples after this time")


def _load(path, a):
    sp = lambda s: [x.strip() for x in s.split(",")] if s else None
    return load(path, cols=sp(a.cols), joints=sp(a.joints), rate=a.rate, topic=a.topic, t0=a.t0, t1=a.t1)


def _cfg(a):
    c = Config()
    for k in ("acc_limit", "jerk_limit"):
        if getattr(a, k, None) is not None:
            setattr(c, k, getattr(a, k))
    return c


def _limits(a):
    if not getattr(a, "urdf", None):
        return {}
    try:
        lim = parse_urdf_limits(open(a.urdf).read())
    except OSError as e:
        raise TrajError(f"cannot read URDF: {e}")
    if not lim:
        raise TrajError(f"no joint limits found in {a.urdf}")
    return lim


def _tip(tr, a):
    if not getattr(a, "fk", False):
        return None
    from trajlens.fk import tip_positions
    return tip_positions(tr, xml=a.xml, tool_length=a.tool_length)


def _jsonable(m):
    return {k: v for k, v in m.items() if k != "_arrays"}


def _fmt(x, f="{:.3f}", unit=""):
    return "n/a" if x is None or x != x else f.format(x) + unit


def report(name, m, findings):
    print(f"{name}: {m['samples']} samples, {m['duration_s']:.2f}s, median step {m['median_dt_s'] * 1e3:.1f} ms")
    print(f"  path length {m['path_length_rad']:.3f} rad, max joint speed {m['max_speed']:.3f} rad/s")
    print(f"  normalized jerk {_fmt(m['normalized_jerk'], '{:.1f}')} (min-jerk reaching = 19), SPARC {_fmt(m['sparc'], '{:.2f}')}")
    if m.get("time_ratio"):
        print(f"  time ratio {m['time_ratio']:.2f} (moving {m['moving_time_s']:.2f}s vs {m['min_time_bound_s']:.2f}s velocity-limit bound)")
    if "tip_path_length_m" in m:
        print(f"  tip path {m['tip_path_length_m']:.3f} m, tip max speed {m['tip_max_speed_mps']:.3f} m/s")
    print(f"  {'joint':<22}{'max vel':>9}{'max acc':>10}{'max jerk':>11}{'range':>8}{'vel use':>9}{'margin':>8}")
    for n, j in m["joints"].items():
        print(f"  {n:<22}{j['max_vel']:>9.3f}{j['max_acc']:>10.2f}{j['max_jerk']:>11.1f}{j['range']:>8.2f}"
              f"{_fmt(j.get('vel_utilization'), '{:.0%}'):>9}{_fmt(j.get('pos_margin'), '{:.0%}'):>8}")
    if findings:
        for f in findings:
            print(f"  {f}")
    else:
        print("  no findings")


def cmd_analyze(a):
    tr = _load(a.file, a)
    lim = _limits(a)
    tip = _tip(tr, a)
    cfg = _cfg(a)
    m = analyze(tr, lim, tip, smooth=a.smooth, jerk_limit=cfg.jerk_limit, acc_limit=cfg.acc_limit)
    f = trajectory_findings(tr, m, cfg, lim)
    report(a.file, m, f)
    if a.out:
        from trajlens.plot import plot_report
        plot_report(tr, m, f, a.out, limits=lim, acc_limit=cfg.acc_limit, jerk_limit=cfg.jerk_limit)
        print(f"wrote {a.out}")
    if a.json:
        out = _jsonable(m)
        out["findings"] = [x.to_dict() for x in f]
        out["source"] = a.file
        json.dump(out, open(a.json, "w"), indent=2)
        print(f"wrote {a.json}")
    return 1 if a.strict and worst_level(f) in ("ERROR", "WARN") else 0


def cmd_compare(a):
    lim = _limits(a)
    items = []
    for p in a.files:
        tr = _load(p, a)
        tip = _tip(tr, a)
        m = analyze(tr, lim, tip, smooth=a.smooth)
        items.append((p, tr, m, trajectory_findings(tr, m, _cfg(a), lim)))
    keys = [("duration_s", "duration [s]", "{:.2f}"), ("path_length_rad", "path [rad]", "{:.3f}"), ("max_speed", "max speed [rad/s]", "{:.3f}"),
            ("max_jerk_norm", "max jerk [rad/s3]", "{:.1f}"), ("normalized_jerk", "normalized jerk", "{:.1f}"), ("sparc", "SPARC", "{:.2f}"),
            ("time_ratio", "time ratio", "{:.2f}"), ("tip_path_length_m", "tip path [m]", "{:.3f}")]
    import os
    lab_of = lambda p: os.path.basename(p)
    w = max(len(lab_of(p)) for p, *_ in items) + 2
    print(f"{'':<20}" + "".join(f"{lab_of(p):>{w}}" for p, *_ in items))
    for k, lab, fm in keys:
        if all(k not in m or m[k] is None for _, _, m, _ in items):
            continue
        print(f"{lab:<20}" + "".join(f"{_fmt(m.get(k), fm):>{w}}" for _, _, m, _ in items))
    print(f"{'findings':<20}" + "".join(f"{len(f):>{w}}" for *_, f in items))
    for p, _, _, f in items:
        for x in f:
            print(f"  {p}: {x}")
    if a.out:
        from trajlens.plot import plot_compare
        plot_compare([(p, tr, m) for p, tr, m, _ in items], a.out)
        print(f"wrote {a.out}")
    if a.json:
        json.dump({p: dict(_jsonable(m), findings=[x.to_dict() for x in f]) for p, _, m, f in items}, open(a.json, "w"), indent=2)
        print(f"wrote {a.json}")
    return 0


def cmd_track(a):
    from trajlens.track import track
    plan, ex = _load(a.planned, a), load(a.executed, joints=None, rate=a.rate, topic=a.topic)
    m, f, al = track(plan, ex, max_lag=a.max_lag, err_warn=a.err_warn, final_warn=a.final_warn)
    print(f"tracking {a.executed} against {a.planned} ({len(m['matched_joints'])} joints)")
    print(f"  lag {m['lag_s'] * 1e3:+.0f} ms, rms error {m['rms_error']:.4f} rad ({m['rms_error_after_lag']:.4f} after removing the lag), "
          f"max {m['max_error']:.4f} rad, final {m['final_error']:.4f} rad")
    for n, j in m["joints"].items():
        print(f"  {n:<22} rms {j['rms_error']:.4f}  max {j['max_error']:.4f}")
    for x in f:
        print(f"  {x}")
    if not f:
        print("  no findings")
    if a.out:
        from trajlens.plot import plot_track
        plot_track(al, m, f, a.out)
        print(f"wrote {a.out}")
    if a.json:
        json.dump(dict(m, findings=[x.to_dict() for x in f]), open(a.json, "w"), indent=2)
        print(f"wrote {a.json}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(prog="trajlens", description="Quality analysis of robot arm joint trajectories.")
    ap.add_argument("--version", action="version", version=f"trajlens {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def limits(p):
        p.add_argument("--urdf", help="URDF file for joint position and velocity limits")
        p.add_argument("--acc-limit", type=float, dest="acc_limit", help="acceleration limit for all joints [rad/s^2]")
        p.add_argument("--jerk-limit", type=float, dest="jerk_limit", help="jerk limit for all joints [rad/s^3]")
        p.add_argument("--smooth", type=int, help="moving-average window (samples) applied before differentiating")
        p.add_argument("--fk", action="store_true", help="tip-space metrics through MuJoCo forward kinematics (UR3)")
        p.add_argument("--xml", help="MuJoCo model for --fk (or set TRAJLENS_UR3_XML)")
        p.add_argument("--tool-length", type=float, default=0.0, dest="tool_length", help="tool length along the site z axis [m]")
        p.add_argument("--out", help="write a PNG report")
        p.add_argument("--json", help="write metrics and findings as JSON")

    p = sub.add_parser("analyze", help="metrics and findings for one trajectory")
    p.add_argument("file")
    _common(p)
    limits(p)
    p.add_argument("--strict", action="store_true", help="exit 1 when there is a warning or error")
    p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("compare", help="side by side metrics for several trajectories")
    p.add_argument("files", nargs="+")
    _common(p)
    limits(p)
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("track", help="planned against executed trajectory")
    p.add_argument("planned")
    p.add_argument("executed")
    _common(p)
    p.add_argument("--max-lag", type=float, default=1.0, dest="max_lag")
    p.add_argument("--err-warn", type=float, default=0.05, dest="err_warn", help="max error that raises TRACKING_ERROR [rad]")
    p.add_argument("--final-warn", type=float, default=0.01, dest="final_warn")
    p.add_argument("--out")
    p.add_argument("--json")
    p.set_defaults(fn=cmd_track)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        return a.fn(a)
    except TrajError as e:
        print(f"trajlens: {e}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
