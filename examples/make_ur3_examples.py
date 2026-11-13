"""Build the UR3 example trajectories in MuJoCo: a smooth move and a jerky move through the same three tool poses,
and the executed motion of the UR3 position actuators following each plan.

Needs mujoco and reachlens (for inverse kinematics). reachlens is not a dependency of trajlens; run this with
    PYTHONPATH=~/reachlens python examples/make_ur3_examples.py
"""
import os
from pathlib import Path

import mujoco
import numpy as np

from reachlens.ik import solve_pose, tool_down
from reachlens.arms import load_arm

OUT = Path(__file__).parent / "data"
RATE = 100.0
T_SEG = 1.5                                     # seconds per segment
WAYPOINTS = [(0.28, -0.20, 0.20), (0.28, 0.00, 0.32), (0.28, 0.20, 0.20)]


def ik_waypoints(model, data, sid, q0):
    qs, seed = [], q0
    for p in WAYPOINTS:
        r = solve_pose(model, data, sid, np.array(p), tool_down(), seeds=[seed])
        if not r.reachable:
            raise SystemExit(f"waypoint {p} not reachable (pos err {r.pos_err:.4f} m)")
        qs.append(r.q[:6].copy())
        seed = r.q
    return qs


def quintic(q0, q1, v0, v1, T, s):
    """C2 quintic Hermite from q0 to q1 with end velocities v0, v1 and zero end accelerations; s in [0, T]."""
    u = s / T
    h00 = 1 - 10 * u**3 + 15 * u**4 - 6 * u**5
    h10 = u - 6 * u**3 + 8 * u**4 - 3 * u**5
    h01 = 10 * u**3 - 15 * u**4 + 6 * u**5
    h11 = -4 * u**3 + 7 * u**4 - 3 * u**5
    return (h00[:, None] * q0 + h10[:, None] * T * v0 + h01[:, None] * q1 + h11[:, None] * T * v1)


def plan_smooth(qs, t):
    vvia = (qs[2] - qs[0]) / (2 * T_SEG)
    z = np.zeros(6)
    out = []
    for k, (a, b, va, vb) in enumerate([(qs[0], qs[1], z, vvia), (qs[1], qs[2], vvia, z)]):
        m = (t >= k * T_SEG) & (t <= (k + 1) * T_SEG + 1e-9)
        out.append((m, quintic(a, b, va, vb, T_SEG, t[m] - k * T_SEG)))
    q = np.zeros((len(t), 6))
    for m, v in out:
        q[m] = v
    return q


def plan_jerky(qs, t):
    """Constant velocity on each segment, instant direction change at the via point: velocity jumps."""
    q = np.zeros((len(t), 6))
    for i, ti in enumerate(t):
        k = min(int(ti // T_SEG), 1)
        u = (ti - k * T_SEG) / T_SEG
        q[i] = qs[k] + u * (qs[k + 1] - qs[k])
    return q


def execute(model, q_plan, t, hold=0.5):
    """Drive the position actuators with the plan (interpolated to the physics step) and record qpos at the plan rate."""
    data = mujoco.MjData(model)
    data.qpos[:6] = q_plan[0]
    data.ctrl[:6] = q_plan[0]
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep
    n = int(round(1.0 / RATE / dt))
    rec = [data.qpos[:6].copy()]
    q_plan = np.vstack([q_plan, np.repeat(q_plan[-1:], int(hold * RATE), axis=0)])     # hold the final pose to settle
    for i in range(1, len(q_plan)):
        for k in range(n):
            u = (k + 1) / n
            data.ctrl[:6] = (1 - u) * q_plan[i - 1] + u * q_plan[i]
            mujoco.mj_step(model, data)
        rec.append(data.qpos[:6].copy())
    return np.array(rec)


def write_csv(path, t, q, names):
    np.savetxt(path, np.column_stack([t, q]), delimiter=",", header="t," + ",".join(names), comments="", fmt="%.6f")


def main():
    model, data, sid, q0 = load_arm("ur3")
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(6)]
    qs = ik_waypoints(model, data, sid, q0)
    t = np.arange(0, 2 * T_SEG + 1e-9, 1.0 / RATE)
    OUT.mkdir(exist_ok=True)
    plans = {"ur3_smooth": plan_smooth(qs, t), "ur3_jerky": plan_jerky(qs, t)}
    for name, q in plans.items():
        write_csv(OUT / f"{name}.csv", t, q, names)
        ex = execute(model, q, t)
        write_csv(OUT / f"{name}_executed.csv", np.arange(len(ex)) / RATE, ex, names)
    print("wrote", sorted(p.name for p in OUT.glob("*.csv")))


if __name__ == "__main__":
    main()
