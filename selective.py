"""Stage two: the label-free rejection gate and its risk-coverage curve."""

import numpy as np
import pandas as pd

from .config import CLAMP, MODE
from .evaluate import gt_fiducials, qtc, r_peaks
from .models import SEG_PRED
from .refinement import apply_refinement


def beat_spread(rec, method, models):
    """Interquartile range of the per-beat QTc within one record."""
    try:
        on, off, _ = SEG_PRED[method](rec, models[method])
        rp = r_peaks(rec)
    except Exception:
        return np.nan
    if len(on) < 2 or len(rp) < 2:
        return np.nan
    fs = rec["fs"]
    rr = np.median(np.diff(rp)) / fs
    if not np.isfinite(rr) or rr <= 0:
        return np.nan
    qt = (np.asarray(off, float) - np.asarray(on, float)) / fs
    qt = qt[(qt > 0.20) & (qt < 0.70)]
    if len(qt) < 2:
        return np.nan
    beats = 1000.0 * qt / np.sqrt(rr)
    return float(np.subtract(*np.percentile(beats, [75, 25])))


def fit_threshold(recs, method, models, percentile=80):
    """Threshold fitted on held-out training data, so coverage is an outcome."""
    s = np.array([beat_spread(r, method, models) for r in recs])
    s = s[np.isfinite(s)]
    return float(np.percentile(s, percentile)), len(s)


def score_frame(recs, method, models, mode=MODE, clamp=CLAMP, formula="bazett"):
    """Per-record refined error and confidence score."""
    rows = []
    for rec in recs:
        try:
            g_on, g_off = gt_fiducials(rec)
            on, off, _ = SEG_PRED[method](rec, models[method])
            rp = r_peaks(rec)
        except Exception:
            continue
        if len(on) == 0 or len(rp) < 2:
            continue
        fs, sig = rec["fs"], np.asarray(rec["sig"], float)
        on2, off2 = apply_refinement(sig, on, off, fs, mode, clamp)
        g = qtc(g_on, g_off, rp, fs, formula)
        v = qtc(on2, off2, rp, fs, formula)
        if not (np.isfinite(g) and np.isfinite(v)):
            continue
        rows.append({"id": rec["id"], "err": v - g,
                     "spread": beat_spread(rec, method, models)})
    G = pd.DataFrame(rows)
    return G[np.isfinite(G["err"]) & np.isfinite(G["spread"])].reset_index(drop=True)


def selective_table(G, thr):
    """Error with and without the gate applied."""
    keep = G["spread"].values <= thr
    e_all, e_keep = G["err"].values, G["err"].values[keep]

    def _row(tag, e, cov, n):
        return {"set": tag, "n": n, "coverage %": cov, "bias": e.mean(),
                "MAE": np.abs(e).mean(), "SD": e.std(ddof=1),
                "over 50ms %": 100 * (np.abs(e) > 50).mean(),
                "within 15ms %": 100 * (np.abs(e) <= 15).mean()}

    return pd.DataFrame([
        _row("all records", e_all, 100.0, len(e_all)),
        _row("spread <= threshold", e_keep, 100 * keep.mean(), int(keep.sum())),
    ]).round(2), keep


def random_control(G, keep, rng, n_rand=2000, n_boot=5000, thr=None):
    """Random rejection of the same count, and a bootstrap over the gain."""
    e_all = G["err"].values
    spread_all = G["spread"].values
    e_keep = e_all[keep]
    rand = np.array([np.abs(e_all[rng.permutation(len(e_all))[:keep.sum()]]).mean()
                     for _ in range(n_rand)])
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        j = rng.integers(0, len(e_all), len(e_all))
        eb, kb = e_all[j], spread_all[j] <= thr
        diffs[i] = (np.abs(eb).mean() - np.abs(eb[kb]).mean()
                    if kb.sum() >= 10 else np.nan)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    gain = np.abs(e_all).mean() - np.abs(e_keep).mean()
    return {"gain": gain, "lo": lo, "hi": hi, "random": rand.mean(),
            "threshold": thr, "coverage": 100 * keep.mean()}


def coverage_curve(G, min_k=40):
    """MAE and failure rate as coverage falls, records ordered by score."""
    e_all = G["err"].values
    order = np.argsort(G["spread"].values)
    e_sort = e_all[order]
    n = len(e_sort)
    cov, mae, o50 = [], [], []
    for k in range(min_k, n + 1):
        a = np.abs(e_sort[:k])
        cov.append(100 * k / n)
        mae.append(a.mean())
        o50.append(100 * (a > 50).mean())
    return np.array(cov), np.array(mae), np.array(o50)
