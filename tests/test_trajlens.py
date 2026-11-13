import json
import math

import numpy as np
import pytest

from trajlens import __version__
from trajlens.cli import main
from trajlens.findings import Config, trajectory_findings
from trajlens.io import Trajectory, TrajError, load
from trajlens.metrics import analyze, derivatives, movements, parse_urdf_limits
from trajlens.track import track

URDF = """<robot name="r">
 <joint name="a" type="revolute"><limit lower="-1" upper="1" velocity="2" effort="1"/></joint>
 <joint name="b" type="continuous"><limit velocity="3" effort="1"/></joint>
 <joint name="fix" type="fixed"/>
</robot>"""


def minjerk(n=401, T=2.0, amp=(1.0, -0.5), t0=0.0):
    t = np.linspace(0, T, n) + t0
    u = (t - t0) / T
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    return Trajectory(t, np.outer(s, amp), ["a", "b"], "synthetic", [])


def write_csv(path, tr):
    np.savetxt(path, np.column_stack([tr.t, tr.q]), delimiter=",", header="t," + ",".join(tr.names), comments="")


# ------------------------------------------------------------------ metrics

def test_parse_urdf_limits():
    lim = parse_urdf_limits(URDF)
    assert set(lim) == {"a", "b"}
    assert lim["a"]["lower"] == -1 and lim["a"]["velocity"] == 2 and not lim["a"]["continuous"]
    assert lim["b"]["continuous"] and lim["b"]["lower"] is None
    assert parse_urdf_limits("not xml") == {}


def test_derivatives_of_a_cubic():
    t = np.linspace(0, 1, 101)
    q = (t**3)[:, None]
    v, a, j = derivatives(t, q)
    assert v[50, 0] == pytest.approx(3 * 0.5**2, abs=1e-3)
    assert a[50, 0] == pytest.approx(6 * 0.5, abs=1e-3)
    assert j[50, 0] == pytest.approx(6.0, abs=1e-2)
    assert np.isnan(j[0, 0]) and np.isnan(j[-1, 0])


def test_minimum_jerk_is_near_the_analytic_cost():
    m = analyze(minjerk(1001))
    assert m["normalized_jerk"] == pytest.approx(math.sqrt(360), rel=0.03)
    assert -1.6 < m["sparc"] < -1.2
    assert m["max_speed"] == pytest.approx(1.875 * math.hypot(1.0, 0.5) / 2.0, rel=0.01)


def test_jerky_motion_scores_worse_than_smooth():
    t = np.linspace(0, 2, 201)
    lin = Trajectory(t, np.outer(np.where(t < 1, t, 1 + 0.2 * (t - 1)), [1.0, 0.5]), ["a", "b"], "x", [])
    assert analyze(lin)["normalized_jerk"] > 10 * analyze(minjerk(201))["normalized_jerk"]


def test_limit_margin_and_time_ratio():
    lim = parse_urdf_limits(URDF)
    tr = minjerk(amp=(0.9, -0.5))
    m = analyze(tr, lim)
    assert m["joints"]["a"]["pos_margin"] == pytest.approx(0.05, abs=1e-3)     # 0.1 left of a 2 rad range
    assert "pos_margin" not in m["joints"]["b"]                                # continuous
    assert m["min_time_bound_s"] == pytest.approx(max(0.9 / 2, 0.5 / 3))
    assert m["time_ratio"] == pytest.approx(2.0 / 0.45, rel=1e-6)


def test_sampling_gap_splits_segments():
    tr = minjerk(201)
    keep = np.r_[0:100, 150:201]
    cut = Trajectory(tr.t[keep], tr.q[keep], tr.names, "x", [])
    m = analyze(cut)
    assert m["segments"] == 2
    assert "SAMPLING_GAP" in [f.code for f in trajectory_findings(cut, m)]


def test_movements_split_at_rest():
    a, b = minjerk(201), minjerk(201, t0=3.0)
    t = np.r_[a.t, b.t]
    q = np.vstack([a.q, b.q + a.q[-1]])
    assert len(movements(t, q)) == 2


# ------------------------------------------------------------------ findings

def codes(tr, **kw):
    lim = parse_urdf_limits(URDF)
    m = analyze(tr, lim, **{k: v for k, v in kw.items() if k in ("acc_limit", "jerk_limit")})
    return [f.code for f in trajectory_findings(tr, m, Config(**kw), lim) if f.level != "INFO"]


def test_clean_trajectory_has_no_findings():
    assert codes(minjerk(amp=(0.5, -0.3))) == []


def test_limit_findings():
    assert "LIMIT_NEAR" in codes(minjerk(amp=(0.99, 0.1)))
    assert "LIMIT_EXCEEDED" in codes(minjerk(amp=(1.2, 0.1)))


def test_velocity_over():
    assert "VEL_OVER" in codes(minjerk(T=0.5, amp=(0.9, 0.1)))


def test_jerk_spike_and_stop_start():
    t = np.linspace(0, 3, 301)
    q = np.where(t < 1.5, t, 1.5 + 4 * (t - 1.5))[:, None] * 0.1
    tr = Trajectory(t, np.column_stack([q, 0 * q]), ["a", "b"], "x", [])
    assert "JERK_SPIKE" in codes(tr)
    a = minjerk(201).q * 0.4
    q = np.vstack([a, np.repeat(a[-1:], 60, axis=0), a + a[-1]])
    stop = Trajectory(np.arange(len(q)) * 0.01, q, ["a", "b"], "x", [])
    assert "STOP_START" in codes(stop)


def test_rest_at_the_ends_is_not_a_stop():
    a = minjerk(201)
    q = np.vstack([np.zeros((100, 2)), a.q, np.repeat(a.q[-1:], 100, axis=0)])
    t = np.arange(len(q)) * 0.01
    assert "STOP_START" not in codes(Trajectory(t, q * 0.4, ["a", "b"], "x", []))


# ------------------------------------------------------------------ tracking

def test_tracking_lag_is_recovered():
    plan = minjerk(401, T=2.0)
    late = minjerk(401, T=2.0, t0=0.12)
    late = Trajectory(late.t, late.q, late.names, "x", [])
    ex_t = np.linspace(0, 2.6, 521)
    u = np.clip((ex_t - 0.12) / 2.0, 0, 1)
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    ex = Trajectory(ex_t, np.outer(s, (1.0, -0.5)), ["a", "b"], "x", [])
    m, f, _ = track(plan, ex)
    assert m["lag_s"] == pytest.approx(0.12, abs=0.005)
    assert m["rms_error_after_lag"] < 0.01
    assert "TRACKING_LAG" in [x.code for x in f]
    assert "TRACKING_FINAL" not in [x.code for x in f]


def test_identical_trajectories_track_perfectly():
    m, f, _ = track(minjerk(), minjerk())
    assert m["lag_s"] == 0 and m["max_error"] < 1e-9 and f == []


def test_tracking_matches_joints_by_name_and_rejects_mismatch():
    plan = minjerk()
    ex = Trajectory(plan.t, plan.q[:, ::-1], plan.names[::-1], "x", [])
    m, f, _ = track(plan, ex)
    assert m["max_error"] < 1e-9
    with pytest.raises(TrajError):
        track(plan, Trajectory(plan.t, plan.q[:, :1] * 0, ["z"], "x", []))


def test_executed_ending_early_is_reported():
    plan = minjerk()
    k = len(plan.t) // 2
    ex = Trajectory(plan.t[:k], plan.q[:k], plan.names, "x", [])
    _, f, _ = track(plan, ex)
    assert "EXEC_ENDS_EARLY" in [x.code for x in f]


# ------------------------------------------------------------------ input formats

def test_csv_with_and_without_time_column(tmp_path):
    tr = minjerk(51)
    write_csv(tmp_path / "a.csv", tr)
    got = load(str(tmp_path / "a.csv"))
    assert got.names == ["a", "b"] and np.allclose(got.t, tr.t) and np.allclose(got.q, tr.q)
    np.savetxt(tmp_path / "n.csv", tr.q, delimiter=",", header="x,y", comments="")
    assert np.allclose(load(str(tmp_path / "n.csv"), rate=50).t, np.arange(51) / 50)


def test_json_joint_trajectory(tmp_path):
    pts = [{"positions": [0.0, 0.1 * i], "time_from_start": {"sec": i // 10, "nanosec": (i % 10) * 100_000_000}} for i in range(30)]
    (tmp_path / "t.json").write_text(json.dumps({"joint_names": ["a", "b"], "points": pts}))
    tr = load(str(tmp_path / "t.json"))
    assert tr.names == ["a", "b"] and tr.t[-1] == pytest.approx(2.9) and tr.q[-1, 1] == pytest.approx(2.9)


def test_json_lines_recording(tmp_path):
    lines = [json.dumps({"type": "joints", "t": 0.1 * i, "pos": {"a": 0.01 * i, "b": 0.0}}) for i in range(30)]
    (tmp_path / "r.jsonl").write_text("\n".join(lines))
    tr = load(str(tmp_path / "r.jsonl"))
    assert tr.q.shape == (30, 2)


def test_bad_inputs_raise_trajerror(tmp_path):
    (tmp_path / "e.csv").write_text("")
    with pytest.raises(TrajError):
        load(str(tmp_path / "e.csv"))
    with pytest.raises(TrajError):
        load(str(tmp_path / "missing.csv"))


# ------------------------------------------------------------------ cli

def test_cli_analyze_writes_png_and_json(tmp_path, capsys):
    write_csv(tmp_path / "a.csv", minjerk(amp=(0.5, -0.3)))
    (tmp_path / "r.urdf").write_text(URDF)
    rc = main(["analyze", str(tmp_path / "a.csv"), "--urdf", str(tmp_path / "r.urdf"), "--out", str(tmp_path / "r.png"),
               "--json", str(tmp_path / "m.json")])
    assert rc == 0
    assert (tmp_path / "r.png").stat().st_size > 5000
    out = json.loads((tmp_path / "m.json").read_text())
    assert out["samples"] == 401 and all(f["level"] == "INFO" for f in out["findings"]) and "_arrays" not in out


def test_cli_strict_exit_code(tmp_path):
    write_csv(tmp_path / "a.csv", minjerk(amp=(1.2, 0.0)))
    (tmp_path / "r.urdf").write_text(URDF)
    args = ["analyze", str(tmp_path / "a.csv"), "--urdf", str(tmp_path / "r.urdf")]
    assert main(args) == 0
    assert main(args + ["--strict"]) == 1


def test_cli_compare_and_track(tmp_path, capsys):
    write_csv(tmp_path / "a.csv", minjerk())
    write_csv(tmp_path / "b.csv", minjerk(amp=(0.5, 0.2)))
    assert main(["compare", str(tmp_path / "a.csv"), str(tmp_path / "b.csv"), "--out", str(tmp_path / "c.png")]) == 0
    assert (tmp_path / "c.png").exists()
    assert main(["track", str(tmp_path / "a.csv"), str(tmp_path / "a.csv"), "--out", str(tmp_path / "t.png")]) == 0
    assert (tmp_path / "t.png").exists()
    assert "no findings" in capsys.readouterr().out


def test_cli_errors_and_version(tmp_path, capsys):
    assert main(["analyze", str(tmp_path / "nope.csv")]) == 2
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert __version__ in capsys.readouterr().out


# ------------------------------------------------------------------ optional forward kinematics

def test_tip_metrics_with_mujoco():
    pytest.importorskip("mujoco")
    from trajlens.fk import find_model
    try:
        find_model()
    except TrajError:
        pytest.skip("UR3 model not available")
    from trajlens.fk import tip_positions
    names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    base = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    t = np.linspace(0, 1, 50)
    q = base + np.outer(np.sin(t), [0.3, 0, 0, 0, 0, 0])
    tr = Trajectory(t, q, names, "x", [])
    tip = tip_positions(tr)
    m = analyze(tr, tip=tip)
    assert m["tip_path_length_m"] > 0.01 and len(m["tip_start"]) == 3


def test_rosbag2_joint_states(tmp_path):
    pytest.importorskip("rosbag2_py")
    pytest.importorskip("sensor_msgs.msg")
    import rosbag2_py
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import JointState
    w = rosbag2_py.SequentialWriter()
    w.open(rosbag2_py.StorageOptions(uri=str(tmp_path / "bag"), storage_id="sqlite3"), rosbag2_py.ConverterOptions("", ""))
    w.create_topic(rosbag2_py.TopicMetadata(name="/joint_states", type="sensor_msgs/msg/JointState", serialization_format="cdr"))
    for i in range(60):
        m = JointState()
        m.name, m.position = ["a", "b"], [0.01 * i, 0.0]
        m.header.stamp.sec, m.header.stamp.nanosec = 5 + i // 50, (i % 50) * 20_000_000
        w.write("/joint_states", serialize_message(m), 10**18 + i * 20_000_000)
    del w                                            # close and flush the bag before reading it
    tr = load(str(tmp_path / "bag"))
    assert tr.names == ["a", "b"] and tr.q.shape == (60, 2) and tr.t[-1] - tr.t[0] == pytest.approx(1.18)
