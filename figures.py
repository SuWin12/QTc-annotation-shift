"""Report figures: black and white, print style, models told apart by hatch."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .config import FIG_DIR, WAVE_COLOR, WAVE_NAME
from .evaluate import FLOOR, gt_fiducials

NAME  = {"unet": "U-Net", "beatssl": "BeatSSL", "physiowave": "PhysioWave"}
ORDER = ("unet", "beatssl", "physiowave")


def print_style():
    plt.style.use("default")
    for k in ("text.color", "axes.labelcolor", "axes.titlecolor", "axes.edgecolor",
              "xtick.color", "ytick.color", "xtick.labelcolor", "ytick.labelcolor"):
        matplotlib.rcParams[k] = "black"
    matplotlib.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.size": 10,
        "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "hatch.linewidth": 0.8,
        "patch.edgecolor": "black"})


def _save(fig, name, dpi=300):
    fig.tight_layout()
    fig.savefig(Path(FIG_DIR) / name, dpi=dpi)
    plt.show()
    print(f"  wrote {name}")


def show_anatomy(rec, fname, seconds=3.0):
    """One annotated beat with the QT span marked."""
    fs = rec["fs"]
    n = int(seconds * fs)
    sig = np.asarray(rec["sig"], float)[:n]
    msk = np.asarray(rec["mask"], int)[:n]
    t = np.arange(n) / fs
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(t, sig, color="black", lw=0.9, zorder=3)
    for cls in (1, 2, 3):
        m = msk == cls
        if m.any():
            ax.fill_between(t, sig.min(), sig.max(), where=m, alpha=0.25,
                            color=WAVE_COLOR[cls], lw=0, label=WAVE_NAME[cls])
    g_on, g_off = gt_fiducials(rec)
    for o, f in zip(g_on, g_off):
        if f < n:
            ax.annotate("", xy=(f / fs, sig.max() * 0.92),
                        xytext=(o / fs, sig.max() * 0.92),
                        arrowprops=dict(arrowstyle="<->", color="black", lw=1.3))
            ax.text((o + f) / 2 / fs, sig.max() * 0.97, "QT", ha="center", fontsize=9)
            break
    ax.set_xlabel("time (s)")
    ax.set_ylabel("lead II (mV)")
    ax.set_title(f"{rec['id']} — cardiologist annotation", fontsize=10)
    ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=8)
    _save(fig, fname, dpi=200)


def show_three(panels, fname, seconds=2.5):
    """One panel per database, cardiologist annotation shown on each."""
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.6 * len(panels)))
    axes = np.atleast_1d(axes)
    for ax, (title, rec) in zip(axes, panels):
        fs = rec["fs"]
        n = int(seconds * fs)
        sig = np.asarray(rec["sig"], float)[:n]
        t = np.arange(n) / fs
        ax.plot(t, sig, color="black", lw=0.9, zorder=3)
        if "mask" in rec and np.any(np.asarray(rec["mask"])[:n]):
            msk = np.asarray(rec["mask"], int)[:n]
            for cls in (1, 2, 3):
                m = msk == cls
                if m.any():
                    ax.fill_between(t, sig.min(), sig.max(), where=m,
                                    alpha=0.25, color=WAVE_COLOR[cls], lw=0)
        else:
            try:
                g_on, g_off = gt_fiducials(rec)
                for x, lbl in ((g_on[0], "QRS onset"), (g_off[0], "T offset")):
                    if x / fs < seconds:
                        ax.axvline(x / fs, color="black", ls="--", lw=1.2)
                        ax.text(x / fs, sig.max(), lbl, fontsize=7, rotation=90,
                                va="top", ha="right")
            except Exception:
                pass
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("mV")
    axes[-1].set_xlabel("time (s)")
    _save(fig, fname, dpi=200)


def fig_curves(seg_hist, fname="fig_1_curves.png"):
    """Training against validation loss, one panel per model."""
    ms = [m for m in ORDER if m in seg_hist]
    if not ms:
        print("  skipped: no training history")
        return
    fig, axes = plt.subplots(1, len(ms), figsize=(4.3 * len(ms), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, ms):
        h = seg_hist[m]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        tr, va = np.array(h["train_loss"]), np.array(h["val_loss"])
        ax.fill_between(ep, tr, va, facecolor="0.88", edgecolor="none")
        ax.plot(ep, tr, color="black", lw=1.7, ls="-",  label="training")
        ax.plot(ep, va, color="black", lw=1.7, ls="--", label="validation")
        ax.set_title(NAME[m], fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_xlim(1, len(ep))
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("cross-entropy loss")
    _save(fig, fname)


def fig_dissociation(ABL, fname="fig_2_dissociation.png"):
    """T-offset error and QTc error for every refinement setting."""
    SHORT = {"none": "none", "tangent +/-20": "tan\n$\\pm$20",
             "tangent +/-30": "tan\n$\\pm$30", "knee +/-20": "knee\n$\\pm$20",
             "knee +/-30": "knee\n$\\pm$30", "both +/-20": "both\n$\\pm$20",
             "both +/-30": "both\n$\\pm$30"}
    ms = [m for m in ORDER if m in set(ABL.model)]
    fig, axes = plt.subplots(1, len(ms), figsize=(4.6 * len(ms), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, ms):
        s_ = ABL[ABL.model == m].reset_index(drop=True)
        x = np.arange(len(s_))
        w = 0.38
        ax.bar(x - w / 2, s_["Toff MAE"], w, facecolor="white",
               edgecolor="black", hatch="///", label="T-offset MAE")
        ax.bar(x + w / 2, s_["QTc MAE"], w, facecolor="0.45",
               edgecolor="black", label="QTc MAE")
        it = int(s_["Toff MAE"].idxmin())
        iq = int(s_["QTc MAE"].idxmin())
        top = max(s_["Toff MAE"].max(), s_["QTc MAE"].max())
        ax.plot([it - w / 2], [s_["Toff MAE"][it] + 1.0], marker="v", ms=9,
                mfc="white", mec="black", mew=1.4, clip_on=False)
        ax.plot([iq + w / 2], [s_["QTc MAE"][iq] + 1.0], marker="v", ms=9,
                mfc="black", mec="black", clip_on=False)
        ax.text(it - w / 2, s_["Toff MAE"][it] + 2.4, "best", ha="center", fontsize=7.5)
        ax.text(iq + w / 2, s_["QTc MAE"][iq] + 2.4, "best", ha="center", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT.get(v, v) for v in s_["setting"]], fontsize=7.5)
        ax.set_title(NAME[m], fontsize=10)
        ax.set_ylim(0, top * 1.22)
    axes[0].set_ylabel("error (ms)  $\\downarrow$")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    _save(fig, fname)


def fig_harm(HARM, fname="fig_3_harm.png"):
    """Change in QTc error when the correction is applied to each database."""
    piv = HARM.pivot(index="model", columns="data", values="change")
    piv = piv.reindex([m for m in ORDER if m in piv.index])
    x = np.arange(len(piv))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.bar(x - w / 2, piv["ISP held out"], w, facecolor="white",
           edgecolor="black", hatch="xxx",
           label="training data — no annotation offset")
    ax.bar(x + w / 2, piv["LUDB unseen"], w, facecolor="0.45",
           edgecolor="black", label="unseen data — offset present")
    ax.axhline(0, color="black", lw=1.5)
    for i, m in enumerate(piv.index):
        for dx, col in ((-w / 2, "ISP held out"), (w / 2, "LUDB unseen")):
            v = piv.loc[m, col]
            ax.text(i + dx, v + (0.9 if v >= 0 else -1.9), f"{v:+.2f}",
                    ha="center", fontsize=8.5)
    lo, hi = piv.values.min(), piv.values.max()
    ax.text(-0.48, hi * 0.88, "correction\nmakes it worse", fontsize=8.5,
            style="italic", va="top")
    ax.text(-0.48, lo * 0.72, "correction\nhelps", fontsize=8.5,
            style="italic", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([NAME[m] for m in piv.index])
    ax.set_ylabel("change in QTc MAE when the correction is applied (ms)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _save(fig, fname)


def fig_waterfall(FINAL, fname="fig_4_waterfall.png"):
    """The two stages applied in turn, and what each acts on."""
    d = FINAL
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(d))
    fills   = ["white", "0.70", "0.40"][:len(d)]
    hatches = ["xxx", "///", ""][:len(d)]
    bars = ax.bar(x, d["MAE"], 0.55, facecolor="white", edgecolor="black")
    for b, fc, hh in zip(bars, fills, hatches):
        b.set_facecolor(fc)
        b.set_hatch(hh)
    for i, r in d.iterrows():
        ax.text(i, r["MAE"] + 0.9, f"{r['MAE']:.2f}", ha="center",
                fontsize=11, fontweight="bold")
        ax.text(i, 0.9, f"{r['coverage %']:.0f}% coverage", ha="center", fontsize=8)
    for i in range(len(d) - 1):
        top = max(d["MAE"].iloc[i], d["MAE"].iloc[i + 1])
        ax.annotate("", xy=(i + 1, d["MAE"].iloc[i + 1] + 3.2),
                    xytext=(i, d["MAE"].iloc[i] + 3.2),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.6))
        ax.text(i + 0.5, top + 4.3,
                f"{d['MAE'].iloc[i + 1] - d['MAE'].iloc[i]:+.2f} ms",
                ha="center", fontsize=9)
    ax.axhline(FLOOR["mae"], color="black", ls=(0, (6, 3)), lw=1.2)
    ax.text(len(d) - 0.55, FLOOR["mae"] + 0.7, "reader agreement",
            fontsize=8, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(["as trained", "+ refinement", "+ rejection gate"][:len(d)],
                       fontsize=9)
    ax.set_ylabel("QTc MAE (ms)  $\\downarrow$")

    w = 0.35
    bx.bar(x - w / 2, d["SD"], w, facecolor="white", edgecolor="black",
           hatch="///", label="SD $\\downarrow$   (scatter)")
    bx.bar(x + w / 2, d["over50"], w, facecolor="0.40", edgecolor="black",
           label="over 50 ms \\% $\\downarrow$   (failures)")
    for i, r in d.iterrows():
        bx.text(i - w / 2, r["SD"] + 0.9, f"{r['SD']:.1f}", ha="center", fontsize=8)
        bx.text(i + w / 2, r["over50"] + 0.9, f"{r['over50']:.1f}",
                ha="center", fontsize=8)
    bx.set_xticks(x)
    bx.set_xticklabels(["as trained", "+ refinement", "+ gate"][:len(d)], fontsize=9)
    bx.set_title("what each stage acts on", fontsize=10)
    bx.legend(frameon=False, fontsize=8)
    _save(fig, fname)


def fig_selective(cov, mae, o50, e_all, e_keep, coverage,
                  fname="fig_5_selective.png"):
    """Risk against coverage, with the operating point fitted on training data."""
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True,
                                 gridspec_kw={"height_ratios": [2.1, 1]})
    ax.plot(cov, mae, color="black", lw=2, label="rejection by per-beat spread")
    ax.axhline(np.abs(e_all).mean(), color="black", lw=1.5,
               label=f"random rejection ({np.abs(e_all).mean():.1f} ms)")
    ax.axhline(FLOOR["mae"], color="black", ls="--", lw=1.2,
               label=f"reader agreement ({FLOOR['mae']:.1f} ms)")
    ax.plot([coverage], [np.abs(e_keep).mean()], "o", ms=9, mfc="white",
            mec="black", mew=1.6,
            label=f"threshold from ISP: {coverage:.0f}%, "
                  f"{np.abs(e_keep).mean():.1f} ms")
    ax.set_ylabel("QTc MAE on retained records (ms)")
    ax.set_ylim(0, max(mae) * 1.18)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    bx.plot(cov, o50, color="0.45", lw=2)
    bx.set_xlabel("coverage: records retained (%)")
    bx.set_ylabel("over 50 ms (%)")
    _save(fig, fname, dpi=200)


def fig_bland_altman(frames, methods, fname="bland_altman.png"):
    """Difference against mean, one row per partition, one column per method."""
    fig, axes = plt.subplots(len(frames), len(methods),
                             figsize=(3.1 * len(methods), 3.1 * len(frames)),
                             squeeze=False)
    for i, (ds, d) in enumerate(frames.items()):
        for j, m in enumerate(methods):
            a = axes[i][j]
            if m not in d.columns:
                a.axis("off")
                continue
            p = d[[m, "gt"]].dropna()
            if len(p) < 3:
                a.axis("off")
                continue
            mean = (p[m] + p["gt"]) / 2
            diff = p[m] - p["gt"]
            b, s = diff.mean(), diff.std(ddof=1)
            a.scatter(mean, diff, s=9, alpha=.5, color="black")
            a.axhline(b, color="black", lw=1.4)
            a.axhline(b + 1.96 * s, color="black", ls="--", lw=1)
            a.axhline(b - 1.96 * s, color="black", ls="--", lw=1)
            a.axhline(0, color="grey", lw=.8, alpha=.6)
            a.set_title(f"{m}\n{b:+.1f} $\\pm$ {1.96 * s:.0f} ms", fontsize=9)
            if j == 0:
                a.set_ylabel(f"{ds}\ndifference (ms)", fontsize=9)
            a.set_xlabel("mean QTc (ms)", fontsize=8)
            a.tick_params(labelsize=7)
    _save(fig, fname, dpi=200)
