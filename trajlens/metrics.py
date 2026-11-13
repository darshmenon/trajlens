"""Trajectory quality metrics: derivatives, smoothness, limit margins, path lengths."""
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

try:                                   # optional C++ core, built by setup.py when pybind11 and a compiler are present
    if os.environ.get("TRAJLENS_NO_CPP"):
        raise ImportError
    from trajlens import _fast
except ImportError:
    _fast = None


def have_cpp():
    return _fast is not None


def grad(f, t, edge_order=1):
    """np.gradient(f, t, axis=0, edge_order=...) for a (n, m) array and non-uniform times, in C++ when available."""
    f = np.asarray(f, dtype=float)
    if _fast is not None and f.ndim == 2 and len(t) >= edge_order + 1:
        return _fast.gradient(np.asarray(t, dtype=float), f, edge_order)
    return np.gradient(f, t, axis=0, edge_order=edge_order)


# ---------------------------------------------------------------- limits

def parse_urdf_limits(urdf_text):
    """Joint limits from URDF text: {joint: {lower, upper, velocity, effort, continuous}} for non-fixed joints."""
    out = {}
    try:
        root = ET.fromstring(urdf_text)
    except ET.ParseError:
        return out
    for j in root.iter("joint"):
        lim = j.find("limit")
        if j.get("type") in ("fixed", "floating", "planar") or lim is None:
            continue

        def g(k):
            v = lim.get(k)
            return float(v) if v not in (None, "") else None
        out[j.get("name")] = dict(lower=g("lower"), upper=g("upper"), velocity=g("velocity"), effort=g("effort"),
                                  continuous=j.get("type") == "continuous")
    return out


# ---------------------------------------------------------------- time segments and derivatives

def segments(t, gap_factor=10.0):
    """Index ranges (start, stop) of runs without a sampling gap. A gap is dt > gap_factor * median dt."""
    dt = np.diff(t)
    if len(dt) == 0:
        return [(0, len(t))]
    cut = np.where(dt > gap_factor * np.median(dt))[0]
    edges = [0] + [int(c) + 1 for c in cut] + [len(t)]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b - a >= 1]


def _smooth(q, window):
    if window is None or window < 3:
        return q
    window |= 1                                              # odd
    pad = window // 2
    qp = np.pad(q, ((pad, pad), (0, 0)), mode="edge")
    k = np.ones(window) / window
    return np.column_stack([np.convolve(qp[:, j], k, mode="valid") for j in range(q.shape[1])])


def derivatives(t, q, gap_factor=10.0, smooth=None):
    """Velocity, acceleration and jerk by finite differences (np.gradient, non-uniform time), per gap-free segment.

    Segment edges use second-order one-sided differences; the very first and last jerk sample of each segment are
    NaN because they are not trustworthy. Returns (v, a, j) shaped like q."""
    v = np.full_like(q, np.nan)
    a = np.full_like(q, np.nan)
    j = np.full_like(q, np.nan)
    for lo, hi in segments(t, gap_factor):
        n = hi - lo
        if n < 3:
            continue
        ts, qs = t[lo:hi], _smooth(q[lo:hi], smooth)
        vs = grad(qs, ts, 2)
        v[lo:hi] = vs
        if n >= 4:
            as_ = grad(vs, ts, 2)
            a[lo:hi] = as_
            if n >= 6:
                js = grad(as_, ts, 2)
                js[[0, -1]] = np.nan          # the outermost jerk samples are third-order one-sided: not trusted
                j[lo:hi] = js
    return v, a, j


# ---------------------------------------------------------------- smoothness

def path_length(t, q, gap_factor=10.0):
    """Joint-space path length (sum of ||dq||, rad) without bridging sampling gaps."""
    total = 0.0
    for lo, hi in segments(t, gap_factor):
        if hi - lo >= 2:
            total += float(np.sum(np.linalg.norm(np.diff(q[lo:hi], axis=0), axis=1)))
    return total


def normalized_jerk(t, q, gap_factor=10.0, smooth=None):
    """Dimensionless jerk cost sqrt(0.5 * integral(|jerk|^2 dt) * T^5 / L^2), over the moving, gap-free part.

    L is the joint-space path length and T the duration. Lower is smoother; a minimum-jerk rest-to-rest move gives
    sqrt(360) = 18.97. Returns NaN if the trajectory does not move."""
    _, _, j = derivatives(t, q, gap_factor, smooth)
    total_t, total_l, integral = 0.0, 0.0, 0.0
    for lo, hi in segments(t, gap_factor):
        if hi - lo < 6:
            continue
        ts = t[lo:hi]
        jj = j[lo:hi]
        m = np.all(np.isfinite(jj), axis=1)
        if m.sum() < 2:
            continue
        j2 = np.sum(jj[m] ** 2, axis=1)
        integral += float(np.trapz(j2, ts[m]))
        total_t += float(ts[-1] - ts[0])
        total_l += float(np.sum(np.linalg.norm(np.diff(q[lo:hi], axis=0), axis=1)))
    if total_l < 1e-9 or total_t <= 0:
        return float("nan")
    return math.sqrt(0.5 * integral * total_t ** 5 / total_l ** 2)


def rest_intervals(t, q, gap_factor=10.0, rest_frac=0.02, rest_min_s=0.1):
    """(t_start, t_end) of rests: speed below rest_frac of the peak speed for at least rest_min_s, per gap-free segment."""
    segs = [(lo, hi) for lo, hi in segments(t, gap_factor) if hi - lo >= 3]
    sps = {s: np.linalg.norm(grad(q[s[0]:s[1]], t[s[0]:s[1]]), axis=1) for s in segs}
    peak = max((sp.max() for sp in sps.values()), default=0.0)
    out = []
    if peak <= 0:
        return out
    for (lo, hi), sp in sps.items():
        still = sp < rest_frac * peak
        i = 0
        while i < len(sp):
            if still[i]:
                k = i
                while k + 1 < len(sp) and still[k + 1]:
                    k += 1
                if t[lo + k] - t[lo + i] >= rest_min_s:
                    out.append((lo + i, lo + k))
                i = k + 1
            else:
                i += 1
    return out


def movements(t, q, gap_factor=10.0, **kw):
    """Index ranges (lo, hi) of separate movements: what lies between rests and sampling gaps (rests trimmed)."""
    rests = rest_intervals(t, q, gap_factor, **kw)
    cuts = sorted(rests)
    res = []
    for lo, hi in segments(t, gap_factor):
        pos = lo
        for a, b in cuts:
            if a >= lo and b < hi:
                if a - pos >= 3:
                    res.append((pos, a + 1))
                pos = b
        if hi - pos >= 3:
            res.append((pos, hi))
    # drop pieces that are entirely still
    keep = []
    for a, b in res:
        if np.ptp(q[a:b], axis=0).max() > 1e-6:
            keep.append((a, b))
    return keep


def sparc(t, q, gap_factor=10.0, smooth=None, fc=10.0, amp_threshold=0.05, pad_level=4):
    """Spectral arc length of the joint-space speed profile (Balasubramanian et al. 2015); closer to 0 is smoother.

    Computed on the longest gap-free segment. Returns NaN if there is no movement."""
    segs = segments(t, gap_factor)
    lo, hi = max(segs, key=lambda s: s[1] - s[0])
    if hi - lo < 8:
        return float("nan")
    ts, qs = t[lo:hi], _smooth(q[lo:hi], smooth)
    speed = np.linalg.norm(grad(qs, ts), axis=1)
    if speed.max() < 1e-9:
        return float("nan")
    fs = (len(ts) - 1) / (ts[-1] - ts[0])
    grid = np.linspace(ts[0], ts[-1], len(ts))
    speed = np.interp(grid, ts, speed)
    nfft = int(2 ** (math.ceil(math.log2(len(speed))) + pad_level))
    mag = np.abs(np.fft.fft(speed, nfft))
    freq = np.arange(nfft) * fs / nfft
    half = nfft // 2
    mag = mag[:half] / mag[0] if mag[0] > 0 else mag[:half]
    freq = freq[:half]
    sel = np.where(freq <= fc)[0]
    sel = sel[mag[sel] >= amp_threshold]
    if len(sel) < 2:
        return float("nan")
    f, m = freq[sel[0]:sel[-1] + 1], mag[sel[0]:sel[-1] + 1]
    return float(-np.sum(np.sqrt(np.diff(f / f[-1]) ** 2 + np.diff(m) ** 2)))


# ---------------------------------------------------------------- the full report

def analyze(tr, limits=None, tip=None, gap_factor=10.0, smooth=None, jerk_limit=None, acc_limit=None):
    """Compute all metrics. limits: {joint: {lower, upper, velocity}}; tip: (N, 3) tip positions or None.

    Returns a plain dict plus the arrays under the private key '_arrays' (v, a, j) for plotting and findings."""
    t, q, names = tr.t, tr.q, tr.names
    v, a, j = derivatives(t, q, gap_factor, smooth)
    limits = limits or {}
    segs = segments(t, gap_factor)
    dt = np.diff(t)
    nz = lambda x: float(np.nanmax(np.abs(x))) if np.isfinite(x).any() else float("nan")
    rms = lambda x: float(np.sqrt(np.nanmean(x ** 2))) if np.isfinite(x).any() else float("nan")
    joints = {}
    for i, n in enumerate(names):
        lim = limits.get(n, {})
        lo, hi = lim.get("lower"), lim.get("upper")
        m = dict(min_pos=float(q[:, i].min()), max_pos=float(q[:, i].max()), range=float(q[:, i].max() - q[:, i].min()),
                 max_vel=nz(v[:, i]), rms_vel=rms(v[:, i]), max_acc=nz(a[:, i]), max_jerk=nz(j[:, i]),
                 path_length=path_length(t, q[:, [i]], gap_factor),
                 vel_limit=lim.get("velocity"), lower=lo, upper=hi)
        if lim.get("velocity"):
            m["vel_utilization"] = m["max_vel"] / lim["velocity"]
        if lo is not None and hi is not None and hi > lo and not lim.get("continuous"):
            span = hi - lo
            m["pos_margin"] = float(min(q[:, i].min() - lo, hi - q[:, i].max()) / span)      # negative = outside
        joints[names[i]] = m
    speed = np.linalg.norm(v, axis=1)
    jn = np.linalg.norm(np.where(np.isnan(j), 0.0, j), axis=1)
    jn[~np.isfinite(j).all(axis=1)] = np.nan
    plen = path_length(t, q, gap_factor)
    # lower bound on the duration from the velocity limits alone (ignores acceleration limits): every joint has to
    # cover its own path at no more than its velocity limit
    bounds = [joints[n]["path_length"] / joints[n]["vel_limit"] for n in names if joints[n].get("vel_limit")]
    moving = float(sum(t[hi - 1] - t[lo] for lo, hi in segs))
    mv = movements(t, q, gap_factor)
    njs = [normalized_jerk(t[a:b], q[a:b], gap_factor, smooth) for a, b in mv]
    njs = [x for x in njs if np.isfinite(x)]
    nj_med = float(np.median(njs)) if njs else float("nan")
    out = dict(
        samples=int(len(t)), duration_s=float(t[-1] - t[0]), moving_time_s=moving,
        median_dt_s=float(np.median(dt)), max_dt_s=float(dt.max()), segments=len(segs),
        path_length_rad=plen, max_speed=nz(speed) if np.isfinite(speed).any() else float("nan"),
        max_jerk_norm=float(np.nanmax(jn)) if np.isfinite(jn).any() else float("nan"),
        rms_jerk_norm=float(np.sqrt(np.nanmean(jn ** 2))) if np.isfinite(jn).any() else float("nan"),
        normalized_jerk=nj_med, movements=len(mv), sparc=sparc(t, q, gap_factor, smooth),
        min_time_bound_s=float(max(bounds)) if bounds and max(bounds) > 0 else None,
        joints=joints,
    )
    out["time_ratio"] = (moving / out["min_time_bound_s"]) if out["min_time_bound_s"] else None
    if tip is not None:
        tv = np.full(len(t), np.nan)
        tl = 0.0
        for lo, hi in segs:
            if hi - lo >= 2:
                tl += float(np.sum(np.linalg.norm(np.diff(tip[lo:hi], axis=0), axis=1)))
            if hi - lo >= 3:
                tv[lo:hi] = np.linalg.norm(grad(tip[lo:hi], t[lo:hi]), axis=1)
        out["tip_path_length_m"] = tl
        out["tip_max_speed_mps"] = float(np.nanmax(tv)) if np.isfinite(tv).any() else float("nan")
        out["tip_start"] = [float(x) for x in tip[0]]
        out["tip_end"] = [float(x) for x in tip[-1]]
    out["_arrays"] = dict(v=v, a=a, j=j, speed=speed, jerk_norm=jn)
    return out
