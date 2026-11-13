"""Findings with codes and hints, computed from the metrics of a trajectory."""
from dataclasses import dataclass, field

import numpy as np

from trajlens.metrics import movements, rest_intervals, segments

ERROR, WARN, INFO = "ERROR", "WARN", "INFO"

HINTS = {
    "LIMIT_EXCEEDED": "a joint leaves its URDF position limits; check the planner's joint limits and the start state",
    "LIMIT_NEAR": "a joint comes within a few percent of a position limit; planners and controllers often misbehave there",
    "VEL_OVER": "a joint is faster than its velocity limit; slow the time parameterization (velocity scaling)",
    "ACC_OVER": "acceleration above the given limit; lower the acceleration scaling or add blending",
    "JERK_OVER": "jerk above the given limit; use a jerk-limited time parameterization or smooth the path",
    "JERK_SPIKE": "a sudden jerk peak, typically a velocity or direction jump at a waypoint; blend or time-parameterize with continuous acceleration",
    "ROUGH": "high normalized jerk overall; the trajectory is noisy or piecewise-linear (try --smooth to see if it is sensor noise)",
    "STOP_START": "the arm comes to rest and restarts inside one trajectory; remove the stop or merge the segments",
    "SAMPLING_GAP": "samples are missing for a long time; derivatives are computed on each side of the gap separately",
    "SLOW_PROFILE": "the trajectory takes much longer than the velocity limits require; raise the velocity scaling",
    "FEW_SAMPLES": "very few samples; derivative-based metrics are unreliable",
    "TRACKING_ERROR": "the executed motion deviates from the plan; check controller gains, payload and contact",
    "TRACKING_LAG": "the executed motion follows the plan late; check controller latency and the trajectory start time",
    "TRACKING_FINAL": "the arm did not settle on the final planned pose",
    "EXEC_ENDS_EARLY": "the executed trajectory stops before the plan does; the goal was cancelled or the recording is cut",
    "CLOCK_SHIFT": "the executed timestamps do not overlap the plan (different clocks); starts were aligned",
    "NO_JOINT_MATCH": "planned and executed share no joint names; matched by column order instead",
}


@dataclass
class Finding:
    level: str
    code: str
    msg: str
    t: tuple = None             # (t_start, t_end) of the flagged region, for plots

    def __str__(self):
        return f"{self.level:5} [{self.code}] {self.msg}"

    def to_dict(self):
        d = {"level": self.level, "code": self.code, "message": self.msg, "hint": HINTS.get(self.code, "")}
        if self.t is not None:
            d["t_start"], d["t_end"] = float(self.t[0]), float(self.t[1])
        return d


@dataclass
class Config:
    near_limit: float = 0.02        # fraction of the position range that counts as "near" a limit
    vel_margin: float = 1.0         # velocity above limit * margin is VEL_OVER
    spike_ratio: float = 6.0        # a jerk peak above ratio * rms jerk is a JERK_SPIKE
    rough_nj: float = 100.0         # normalized jerk above this is ROUGH (minimum-jerk reaching is about 19)
    stop_speed_frac: float = 0.02   # below this fraction of the peak speed counts as at rest
    stop_min_s: float = 0.1         # a rest this long between two movements is a STOP_START
    gap_factor: float = 10.0        # dt above this many median dts is a SAMPLING_GAP
    slow_ratio: float = 3.0         # duration / velocity-limit bound above this is SLOW_PROFILE
    few_samples: int = 20
    acc_limit: float = None         # rad/s^2, same for all joints
    jerk_limit: float = None        # rad/s^3, same for all joints


def _regions(mask, t, merge=0.0):
    """(t_start, t_end) of consecutive True runs in mask, merging runs closer than `merge` seconds."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            k = i
            while k + 1 < n and mask[k + 1]:
                k += 1
            out.append([t[i], t[k]])
            i = k + 1
        else:
            i += 1
    merged = []
    for r in out:
        if merged and r[0] - merged[-1][1] <= merge:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [tuple(r) for r in merged]


def trajectory_findings(tr, m, cfg=Config(), limits=None):
    """Findings for one trajectory, given the metrics dict m from metrics.analyze."""
    f = []
    t, q, names = tr.t, tr.q, tr.names
    arr = m["_arrays"]
    if m["samples"] < cfg.few_samples:
        f.append(Finding(WARN, "FEW_SAMPLES", f"only {m['samples']} samples"))
    # sampling gaps
    segs = segments(t, cfg.gap_factor)
    for (a0, a1), (b0, b1) in zip(segs[:-1], segs[1:]):
        f.append(Finding(WARN, "SAMPLING_GAP", f"no samples for {t[b0] - t[a1 - 1]:.2f}s from t={t[a1 - 1]:.2f}s "
                         f"(median step {m['median_dt_s'] * 1e3:.0f} ms)", (t[a1 - 1], t[b0])))
    # limits
    for i, n in enumerate(names):
        jm = m["joints"][n]
        if "pos_margin" in jm:
            lo, hi = jm["lower"], jm["upper"]
            if jm["pos_margin"] < 0:
                bad = (q[:, i] < lo) | (q[:, i] > hi)
                r = _regions(bad, t)
                f.append(Finding(ERROR, "LIMIT_EXCEEDED", f"{n} reaches {jm['min_pos'] if q[:, i].min() < lo else jm['max_pos']:.3f} "
                                 f"outside [{lo:.3f}, {hi:.3f}]", r[0] if r else None))
            elif jm["pos_margin"] < cfg.near_limit:
                span = hi - lo
                near = (q[:, i] < lo + cfg.near_limit * span) | (q[:, i] > hi - cfg.near_limit * span)
                r = _regions(near, t)
                f.append(Finding(WARN, "LIMIT_NEAR", f"{n} comes within {max(jm['pos_margin'], 0):.1%} of its range from a "
                                 f"limit [{lo:.3f}, {hi:.3f}]", r[0] if r else None))
        if jm.get("vel_limit") and jm["max_vel"] > jm["vel_limit"] * cfg.vel_margin:
            over = np.abs(arr["v"][:, i]) > jm["vel_limit"] * cfg.vel_margin
            r = _regions(np.nan_to_num(over, nan=False).astype(bool), t)
            f.append(Finding(ERROR, "VEL_OVER", f"{n} reaches {jm['max_vel']:.2f} rad/s, limit {jm['vel_limit']:.2f}",
                             r[0] if r else None))
        if cfg.acc_limit and jm["max_acc"] > cfg.acc_limit:
            f.append(Finding(ERROR, "ACC_OVER", f"{n} reaches {jm['max_acc']:.2f} rad/s^2, limit {cfg.acc_limit:.2f}"))
        if cfg.jerk_limit and jm["max_jerk"] > cfg.jerk_limit:
            f.append(Finding(ERROR, "JERK_OVER", f"{n} reaches {jm['max_jerk']:.1f} rad/s^3, limit {cfg.jerk_limit:.1f}"))
    mv_ranges = movements(t, q, cfg.gap_factor, rest_frac=cfg.stop_speed_frac, rest_min_s=cfg.stop_min_s)
    # jerk spikes: peaks far above the rms jerk (a smooth profile peaks at 2 to 3 times rms)
    jn = arr["jerk_norm"]
    if np.isfinite(jn).any() and m["rms_jerk_norm"] > 0:
        thr = cfg.spike_ratio * m["rms_jerk_norm"]
        spikes = _regions(np.nan_to_num(jn, nan=0.0) > thr, t, merge=3 * m["median_dt_s"])
        if spikes:
            worst = max(spikes, key=lambda r: np.nanmax(jn[(t >= r[0]) & (t <= r[1])]))
            peak = float(np.nanmax(jn[(t >= worst[0]) & (t <= worst[1])]))
            f.append(Finding(WARN, "JERK_SPIKE", f"{len(spikes)} jerk spike(s), worst {peak:.0f} rad/s^3 at t={worst[0]:.2f}s "
                             f"({peak / m['rms_jerk_norm']:.0f}x the rms jerk)", worst))
            for r in spikes:
                if r != worst:
                    f.append(Finding(INFO, "JERK_SPIKE", f"another spike at t={r[0]:.2f}s", r))
    nj = m["normalized_jerk"]
    if np.isfinite(nj) and nj > cfg.rough_nj:
        f.append(Finding(WARN, "ROUGH", f"normalized jerk {nj:.0f} (minimum-jerk reaching is about 19)"))
    # rests between two movements (rests at the very start or end are not a problem)
    for a0, b0 in rest_intervals(t, q, cfg.gap_factor, cfg.stop_speed_frac, cfg.stop_min_s):
        if any(x < a0 for x, _ in mv_ranges) and any(y > b0 for _, y in mv_ranges):
            f.append(Finding(WARN, "STOP_START", f"at rest for {t[b0] - t[a0]:.2f}s from t={t[a0]:.2f}s between two movements",
                             (t[a0], t[b0])))
    if m.get("time_ratio") and m["time_ratio"] > cfg.slow_ratio:
        f.append(Finding(INFO, "SLOW_PROFILE", f"moving time {m['moving_time_s']:.2f}s is {m['time_ratio']:.1f}x the "
                         f"velocity-limit bound of {m['min_time_bound_s']:.2f}s"))
    return f


def worst_level(findings):
    order = {ERROR: 2, WARN: 1, INFO: 0}
    return max((x.level for x in findings), key=lambda l: order[l], default=None)
