"""Tracking error and lag between a planned and an executed trajectory."""
import numpy as np

from trajlens.findings import ERROR, INFO, WARN, Finding
from trajlens.io import TrajError


def match_joints(plan, ex):
    """Column indices (in plan, in ex) of joints present in both, by name; falls back to order when no names match."""
    common = [n for n in plan.names if n in ex.names]
    if common:
        return [plan.names.index(n) for n in common], [ex.names.index(n) for n in common], common, False
    if len(plan.names) == len(ex.names):
        return list(range(len(plan.names))), list(range(len(ex.names))), list(plan.names), True
    raise TrajError(f"planned joints {plan.names} and executed joints {ex.names} do not match")


def _resample(t, q, grid):
    return np.column_stack([np.interp(grid, t, q[:, j]) for j in range(q.shape[1])])


def estimate_lag(t, p, e, max_lag=1.0, step=0.001, et=None):
    """Time shift s (seconds) that minimizes the rms of e(t + s) - p(t); positive means the executed motion is late."""
    et = t if et is None else et
    best, best_s = None, 0.0
    n = int(round(max_lag / step))
    for k in range(-n, n + 1):
        s = round(k * step, 9)
        tt = t + s
        ok = (tt >= et[0]) & (tt <= et[-1])
        if ok.sum() < max(10, 0.5 * len(t)):
            continue
        err = np.sqrt(np.mean((_resample(et, e, tt[ok]) - p[ok]) ** 2))
        if best is None or err < best:
            best, best_s = err, float(s)
    return best_s


def track(plan, ex, max_lag=1.0, lag_warn=0.05, err_warn=0.05, final_warn=0.01):
    """Compare an executed trajectory with its plan. Returns (metrics dict, findings, aligned arrays for plotting)."""
    ip, ie, names, by_order = match_joints(plan, ex)
    findings = []
    if by_order:
        findings.append(Finding(INFO, "NO_JOINT_MATCH", "planned and executed share no joint names; matched by column order"))
    t0, t1 = plan.t[0], plan.t[-1]
    # the executed clock may start at another time (a recording that begins at 0): align starts when the spans are
    # far apart, never when they already overlap
    shift = 0.0
    if ex.t[0] > t1 or ex.t[-1] < t0:
        shift = t0 - ex.t[0]
        findings.append(Finding(INFO, "CLOCK_SHIFT",
                                f"executed clock does not overlap the plan; shifted by {shift:+.2f}s so both start together"))
    et = ex.t + shift
    grid = plan.t
    P = plan.q[:, ip]
    Eq = ex.q[:, ie]
    covered = (grid >= et[0]) & (grid <= et[-1])
    if covered.sum() < 3:
        raise TrajError("executed trajectory covers (almost) none of the planned time span")
    lag = estimate_lag(grid[covered], P[covered], Eq, max_lag=max_lag, et=et)
    # error without any shift
    E = _resample(et, Eq, grid)
    err = (E - P)[covered]
    joints = {}
    for k, n in enumerate(names):
        joints[n] = dict(rms_error=float(np.sqrt(np.mean(err[:, k] ** 2))), max_error=float(np.max(np.abs(err[:, k]))))
    # error after removing the estimated lag
    tt = grid[covered] + lag
    ok = (tt >= et[0]) & (tt <= et[-1])
    after = _resample(et, Eq, tt[ok]) - P[covered][ok]
    # final error: the last executed sample against the final planned pose (only meaningful when the record reaches the end)
    final_err = float(np.max(np.abs(Eq[-1] - P[-1])))
    m = dict(
        lag_s=lag, rms_error=float(np.sqrt(np.mean(err ** 2))), max_error=float(np.max(np.abs(err))),
        rms_error_after_lag=float(np.sqrt(np.mean(after ** 2))) if len(after) else float("nan"),
        final_error=final_err, planned_duration_s=float(t1 - t0), executed_duration_s=float(et[-1] - et[0]),
        executed_coverage=float(min(1.0, (min(et[-1], t1) - max(et[0], t0)) / (t1 - t0))) if t1 > t0 else 1.0,
        joints=joints, matched_joints=names,
    )
    if lag > lag_warn:
        findings.append(Finding(WARN, "TRACKING_LAG", f"executed motion lags the plan by {lag * 1e3:.0f} ms "
                                f"(rms error {m['rms_error']:.4f} rad, {m['rms_error_after_lag']:.4f} rad without the lag)"))
    elif lag < -lag_warn:
        findings.append(Finding(WARN, "TRACKING_LAG", f"executed motion leads the plan by {-lag * 1e3:.0f} ms; "
                                "check which clock stamps the plan"))
    if m["max_error"] > err_warn:
        k = int(np.argmax(np.max(np.abs(err), axis=0)))
        tm = float(grid[covered][int(np.argmax(np.abs(err[:, k])))])
        findings.append(Finding(WARN, "TRACKING_ERROR", f"{names[k]} deviates by {m['max_error']:.3f} rad at t={tm:.2f}s "
                                f"(limit {err_warn:.3f})", (tm, tm)))
    if final_err > final_warn and et[-1] >= t1 - 1e-9:
        findings.append(Finding(ERROR, "TRACKING_FINAL", f"final pose is off by {final_err:.4f} rad (limit {final_warn:.4f})"))
    if m["executed_coverage"] < 0.95:
        findings.append(Finding(WARN, "EXEC_ENDS_EARLY", f"executed record covers {m['executed_coverage']:.0%} of the planned time"))
    return m, findings, dict(t=grid, plan=P, ex=E, names=names, covered=covered, err=E - P)
