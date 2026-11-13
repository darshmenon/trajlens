"""Matplotlib reports: positions, velocities, accelerations and jerk with limit lines and flagged regions."""
import numpy as np

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#7f7f7f"]
LEVEL_COLOR = {"ERROR": "#d62728", "WARN": "#ff7f0e", "INFO": "#7f7f7f"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _limit_line(ax, val, **kw):
    """Draw a limit line unless it is far outside the data (it would flatten the curves)."""
    lo, hi = ax.get_ylim()
    span = hi - lo
    if lo - 5 * span <= val <= hi + 5 * span:
        ax.axhline(val, **kw)


def _shade(ax, findings):
    for f in findings:
        if f.t is not None and f.level != "INFO" and f.t[1] >= f.t[0]:
            ax.axvspan(f.t[0], max(f.t[1], f.t[0] + 1e-3), color=LEVEL_COLOR[f.level], alpha=0.18, lw=0)


def plot_report(tr, m, findings, path, title=None, limits=None, acc_limit=None, jerk_limit=None):
    plt = _plt()
    t, names, arr = tr.t, tr.names, m["_arrays"]
    limits = limits or {}
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    rows = [("position [rad]", tr.q), ("velocity [rad/s]", arr["v"]), ("acceleration [rad/s²]", arr["a"]), ("jerk [rad/s³]", arr["j"])]
    for ax, (lab, y) in zip(axes, rows):
        for i, n in enumerate(names):
            ax.plot(t, y[:, i], lw=1.1, color=COLORS[i % len(COLORS)], label=n)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25)
        _shade(ax, findings)
    for i, n in enumerate(names):
        lim = limits.get(n, {})
        c = COLORS[i % len(COLORS)]
        for key, ax, sgn in (("lower", axes[0], 1), ("upper", axes[0], 1)):
            if lim.get(key) is not None and not lim.get("continuous"):
                _limit_line(ax, lim[key], color=c, ls="--", lw=0.8, alpha=0.7)
        if lim.get("velocity"):
            for s in (1, -1):
                _limit_line(axes[1], s * lim["velocity"], color=c, ls="--", lw=0.8, alpha=0.7)
    if acc_limit:
        for s in (1, -1):
            axes[2].axhline(s * acc_limit, color="k", ls="--", lw=0.8)
    if jerk_limit:
        for s in (1, -1):
            axes[3].axhline(s * jerk_limit, color="k", ls="--", lw=0.8)
    axes[0].legend(ncol=min(6, len(names)), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.28), frameon=False)
    axes[-1].set_xlabel("time [s]")
    nj = m["normalized_jerk"]
    sub = f"normalized jerk {nj:.0f} · SPARC {m['sparc']:.2f} · {len(findings)} finding(s)" if np.isfinite(nj) else f"{len(findings)} finding(s)"
    fig.suptitle(f"{title or tr.source}\n{sub}", y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_compare(items, path):
    """items: [(label, Trajectory, metrics)]. Overlays speed and jerk norm of each, one color per trajectory."""
    plt = _plt()
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=False)
    for k, (label, tr, m) in enumerate(items):
        c = COLORS[k % len(COLORS)]
        t = tr.t - tr.t[0]
        axes[0].plot(t, np.linalg.norm(tr.q - tr.q[0], axis=1), color=c, lw=1.3, label=label)
        axes[1].plot(t, m["_arrays"]["speed"], color=c, lw=1.2, label=label)
        axes[2].plot(t, m["_arrays"]["jerk_norm"], color=c, lw=1.0, label=label)
    for ax, lab in zip(axes, ("distance from start [rad]", "joint speed norm [rad/s]", "jerk norm [rad/s³]")):
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[2].set_xlabel("time since start [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_track(al, m, findings, path, title="tracking"):
    plt = _plt()
    t, names = al["t"], al["names"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for i, n in enumerate(names):
        c = COLORS[i % len(COLORS)]
        axes[0].plot(t, al["plan"][:, i], color=c, lw=1.4, label=n)
        axes[0].plot(t, al["ex"][:, i], color=c, lw=1.0, ls="--")
        axes[1].plot(t, al["err"][:, i], color=c, lw=1.0, label=n)
    axes[0].set_ylabel("position [rad]")
    axes[1].set_ylabel("executed − planned [rad]")
    axes[1].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(alpha=0.25)
        _shade(ax, findings)
    axes[0].legend(ncol=min(6, len(names)), fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.3), frameon=False,
                   title="solid = plan, dashed = executed", title_fontsize=7)
    fig.suptitle(f"{title}\nlag {m['lag_s'] * 1e3:.0f} ms · rms {m['rms_error']:.4f} rad · max {m['max_error']:.4f} rad", y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)
