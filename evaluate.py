"""Ground truth, QTc arithmetic, beat matching, metrics and the evaluator."""

import warnings

import neurokit2 as nk
import numpy as np
import pandas as pd

from .config import CLAMP, QT_MAX, QT_MIN, RR_MAX, RR_MIN, THRESH
from .models import SEG_PRED, predict_qtcnet, predict_wavelet
from .refinement import apply_refinement

FORMULAE = {
    "bazett":     lambda qt, rr: qt / np.sqrt(rr),
    "fridericia": lambda qt, rr: qt / np.cbrt(rr),
    "framingham": lambda qt, rr: qt + 154 * (1 - rr),
    "hodges":     lambda qt, rr: qt + 1.75 * (60 / rr - 60),
    "rautaharju": lambda qt, rr: qt * (120 + 60 / rr) / 180,
}

FLOOR = {"sd": 9.57, "mae": 9.57 * np.sqrt(2 / np.pi)}


def set_floor(sd):
    """Set the label noise floor from the measured PTB reader agreement."""
    FLOOR["sd"] = float(sd)
    FLOOR["mae"] = float(sd) * np.sqrt(2 / np.pi)
    return FLOOR


def gt_fiducials(rec):
    """(qrs_onsets, t_offsets) in samples, one pair per beat."""
    if "ann" in rec:
        s, y = np.asarray(rec["ann"].sample), list(rec["ann"].symbol)
        on, off = [], []
        for i, sym in enumerate(y):
            if sym != "N":
                continue
            j = i - 1
            while j >= 0 and y[j] != "(":
                j -= 1
            k = i + 1
            while k < len(y) and y[k] != "t":
                k += 1
            if j < 0 or k >= len(y):
                continue
            m = k + 1
            while m < len(y) and y[m] != ")":
                m += 1
            if m >= len(y):
                continue
            on.append(s[j])
            off.append(s[m])
        return np.asarray(on, float), np.asarray(off, float)
    if "waves" in rec:
        w = rec["waves"]
        on, off = [], []
        for i, (cls, o, _) in enumerate(w):
            if cls != 1:
                continue
            nxt = [b for c2, _, b in w[i + 1:] if c2 == 2]
            if nxt:
                on.append(o)
                off.append(nxt[0])
        return np.asarray(on, float), np.asarray(off, float)
    if "q_ref" in rec and "t_ref" in rec:
        return np.array([rec["q_ref"]], float), np.array([rec["t_ref"]], float)
    raise KeyError(f"{rec.get('id', '?')}: no annotation")


def r_peaks(rec):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = nk.ecg_clean(np.asarray(rec["sig"], float), sampling_rate=rec["fs"])
        _, info = nk.ecg_peaks(clean, sampling_rate=rec["fs"])
    return np.asarray(info["ECG_R_Peaks"], float)


def assign_rr(onsets, rpeaks, fs):
    """RR interval immediately preceding each beat, in seconds."""
    rr = np.full(len(onsets), np.nan)
    if len(rpeaks) < 2:
        return rr
    ri = np.diff(rpeaks) / fs
    for k, o in enumerate(onsets):
        j = np.searchsorted(rpeaks, o)
        if 1 <= j <= len(ri):
            rr[k] = ri[j - 1]
    return rr


def qtc(onsets, offsets, rpeaks, fs, formula="bazett", min_beats=2):
    """Record median corrected interval, in milliseconds."""
    n = min(len(onsets), len(offsets))
    if n == 0:
        return np.nan
    qt = (np.asarray(offsets)[:n] - np.asarray(onsets)[:n]) / fs * 1000
    rr = assign_rr(np.asarray(onsets)[:n], rpeaks, fs)
    if not np.isfinite(rr).any() and len(rpeaks) > 1:
        rr = np.full(n, np.median(np.diff(rpeaks) / fs))
    ok = (np.isfinite(qt) & np.isfinite(rr) & (qt >= QT_MIN) & (qt <= QT_MAX)
          & (rr >= RR_MIN) & (rr <= RR_MAX))
    if ok.sum() < min_beats:
        return np.nan
    return float(np.median(FORMULAE[formula](qt[ok], rr[ok])))


def match(gt, pred, fs, tol_ms=150):
    """Greedy nearest-unused matching. Returns (errors in ms, tp, fn)."""
    gt, pred = np.asarray(gt, float), np.asarray(pred, float)
    if len(gt) == 0:
        return [], 0, 0
    if len(pred) == 0:
        return [], 0, len(gt)
    tol = tol_ms * fs / 1000.0
    errs, used = [], set()
    for g in gt:
        cand = [(abs(p - g), j) for j, p in enumerate(pred) if j not in used]
        if not cand:
            break
        d, j = min(cand)
        if d <= tol:
            used.add(j)
            errs.append((pred[j] - g) / fs * 1000)
    return errs, len(errs), len(gt) - len(errs)


def qtc_errors(recs, method, models, refine=None, clamp=CLAMP, shift_ms=0.0,
               formula="bazett", verbose=False):
    """Per-record (predicted, reference) QTc for one method."""
    P, R = [], []
    skipped = {"no reference": 0, "no prediction": 0, "not finite": 0,
               "wrong formula": 0}

    for rec in recs:
        try:
            g_on, g_off = gt_fiducials(rec)
            rp = r_peaks(rec)
        except Exception:
            skipped["no reference"] += 1
            continue
        if len(g_on) == 0 or len(rp) < 2:
            skipped["no reference"] += 1
            continue
        fs = rec["fs"]
        g = qtc(g_on, g_off, rp, fs, formula)

        if method == "qtcnet":
            if formula != "bazett":
                skipped["wrong formula"] += 1
                continue
            try:
                v = predict_qtcnet(rec)
            except Exception as e:
                skipped["no prediction"] += 1
                if verbose:
                    print(f"    qtcnet failed on {rec.get('id')}: {e}")
                continue
            if np.isfinite(g) and np.isfinite(v):
                P.append(v)
                R.append(g)
            else:
                skipped["not finite"] += 1
            continue

        try:
            if method in ("dwt", "cwt"):
                on, off, _ = predict_wavelet(rec, method)
            else:
                on, off, _ = SEG_PRED[method](rec, models[method])
        except Exception as e:
            skipped["no prediction"] += 1
            if verbose:
                print(f"    {method} failed on {rec.get('id')}: {e}")
            continue
        if len(on) == 0:
            skipped["no prediction"] += 1
            continue

        sig = np.asarray(rec["sig"], float)
        off = np.asarray(off, float).copy()
        if shift_ms:
            off = off + shift_ms * fs / 1000.0
        if refine:
            on, off = apply_refinement(sig, on, off, fs, refine, clamp)
        v = qtc(on, off, rp, fs, formula)
        if np.isfinite(g) and np.isfinite(v):
            P.append(v)
            R.append(g)
        else:
            skipped["not finite"] += 1

    if len(P) == 0:
        print(f"  WARNING: {method} produced 0 usable records out of {len(recs)}. "
              "Skipped: " + ", ".join(f"{k}={v}" for k, v in skipped.items() if v))
    return np.asarray(P), np.asarray(R)


def toffset_errors(recs, method, models, refine=None, clamp=CLAMP):
    """Signed T-offset errors in ms, matched beat by beat."""
    d = []
    for rec in recs:
        try:
            _, g_off = gt_fiducials(rec)
            if method in ("dwt", "cwt"):
                on, off, _ = predict_wavelet(rec, method)
            else:
                on, off, _ = SEG_PRED[method](rec, models[method])
        except Exception:
            continue
        if len(on) == 0:
            continue
        fs, sig = rec["fs"], np.asarray(rec["sig"], float)
        if refine:
            on, off = apply_refinement(sig, on, off, fs, refine, clamp)
        b, _, _ = match(g_off, np.asarray(off, float), fs, tol_ms=150)
        d.extend(b)
    return np.asarray(d)


def summarise(P, R, n_total=None):
    """Centre, spread, tail, coverage and decision metrics for one method."""
    e, a = P - R, np.abs(P - R)
    pos, ref = P > THRESH, R > THRESH
    tp = np.sum(pos & ref)
    fp = np.sum(pos & ~ref)
    tn = np.sum(~pos & ~ref)
    fn = np.sum(~pos & ref)
    return {"n": len(e), "bias": e.mean(), "MAE": a.mean(),
            "median AE": np.median(a), "SD": e.std(ddof=1),
            "P90": np.percentile(a, 90), "over50": 100 * (a > 50).mean(),
            "within15": 100 * (a <= 15).mean(),
            "usable": 100 * len(e) / n_total if n_total else np.nan,
            "slope": np.polyfit(R, P, 1)[0] if len(R) > 2 else np.nan,
            "sens": 100 * tp / max(tp + fn, 1),
            "spec": 100 * tn / max(tn + fp, 1),
            "acc": 100 * (tp + tn) / max(len(pos), 1),
            "MAE/floor": a.mean() / FLOOR["mae"]}


def predict_seg_any(rec, method, models, refine=None, max_shift_ms=30):
    fs, sig = rec["fs"], np.asarray(rec["sig"], float)
    if method in ("dwt", "cwt"):
        return predict_wavelet(rec, method)
    on, off, _ = SEG_PRED[method](rec, models[method])
    rp = r_peaks(rec)
    if refine:
        on, off = apply_refinement(sig, on, off, fs, refine, max_shift_ms)
    return on, off, rp


def evaluate_seg(recs, models, *, name, refine=None, max_shift_ms=30,
                 methods=("dwt", "cwt", "unet", "beatssl", "physiowave"),
                 include_qtcnet=True, formula="bazett",
                 tol_on=6.5, tol_off=30.6, quiet=True):
    """-> (fiducial table, QTc table, per-record frame)"""
    fid  = {m: {"on": [], "off": [], "tp": 0, "fn": 0} for m in methods}
    rows, seen = [], set()

    for rec in recs:
        fs = rec["fs"]
        try:
            gt_on, gt_off = gt_fiducials(rec)
        except Exception:
            continue
        if len(gt_on) == 0:
            continue
        gt = qtc(gt_on, gt_off, r_peaks(rec), fs, formula)
        if not np.isfinite(gt):
            continue

        row = {"dataset": name, "id": rec["id"], "gt": gt,
               "sex": rec.get("sex"), "hr": rec.get("hr", np.nan)}

        for m in methods:
            if m in SEG_PRED and m not in models:
                row[m] = np.nan
                continue
            try:
                on, off, rp = predict_seg_any(rec, m, models, refine, max_shift_ms)
                row[m] = qtc(on, off, rp, fs, formula)
            except Exception as e:
                if m not in seen and not quiet:
                    print(f"  {m}: {type(e).__name__}: {e}")
                    seen.add(m)
                on = off = np.array([])
                row[m] = np.nan
            e1, tp1, fn1 = match(gt_on,  on,  fs, tol_ms=150)
            e2, tp2, fn2 = match(gt_off, off, fs, tol_ms=150)
            fid[m]["on"].extend(e1)
            fid[m]["off"].extend(e2)
            fid[m]["tp"] += tp1 + tp2
            fid[m]["fn"] += fn1 + fn2

        if include_qtcnet:
            try:
                row["qtcnet"] = predict_qtcnet(rec)
            except Exception:
                row["qtcnet"] = np.nan
        rows.append(row)

    f_rows = []
    for m, r in fid.items():
        for pt, errs, tol in [("QRS onset", r["on"], tol_on),
                              ("T offset",  r["off"], tol_off)]:
            e = np.asarray(errs, float)
            if e.size == 0:
                continue
            f_rows.append({"dataset": name, "refine": refine or "none", "method": m,
                           "point": pt, "n": int(e.size), "bias": e.mean(),
                           "SD": e.std(ddof=1) if e.size > 1 else 0.0,
                           "MAE": np.abs(e).mean(),
                           "within_tol_%": 100 * np.mean(np.abs(e) <= tol),
                           "sensitivity_%": (100 * r["tp"] / (r["tp"] + r["fn"])
                                             if r["tp"] + r["fn"] else 0.0)})
    fdf = pd.DataFrame(f_rows).round(2)

    df = pd.DataFrame(rows)
    all_m = list(methods) + (["qtcnet"] if include_qtcnet else [])
    q_rows = []
    for m in all_m:
        if m not in df.columns:
            continue
        d = df[[m, "gt"]].dropna()
        if len(d) < 3:
            continue
        e = (d[m] - d["gt"]).values
        q_rows.append({"dataset": name, "refine": refine or "none", "method": m,
                       "n": len(d), "coverage_%": 100 * len(d) / len(df),
                       "bias": e.mean(), "SD": e.std(ddof=1),
                       "MAE": np.abs(e).mean(),
                       "median_AE": float(np.median(np.abs(e))),
                       "p90_AE": float(np.percentile(np.abs(e), 90)),
                       "corr": float(np.corrcoef(d[m], d["gt"])[0, 1]),
                       "IEC": ("PASS" if abs(e.mean()) <= 25 and e.std(ddof=1) <= 30
                               else "FAIL")})
    qdf = pd.DataFrame(q_rows).sort_values("MAE").round(2) if q_rows else pd.DataFrame()

    print(f"[{name}] {len(df)} records | " +
          " | ".join(f"{m} {int(df[m].notna().sum())}"
                     for m in ["gt"] + all_m if m in df))
    return fdf, qdf, df


def boot_ci(err, n=2000, seed=0):
    """MAE with a bootstrap 95 per cent interval over records."""
    rng = np.random.default_rng(seed)
    e = np.asarray(err, float)
    e = e[np.isfinite(e)]
    if len(e) < 5:
        return np.nan, np.nan, np.nan
    s = np.abs(e[rng.integers(0, len(e), (n, len(e)))]).mean(1)
    return np.abs(e).mean(), *np.percentile(s, [2.5, 97.5])


def paired_boot(d, m, best, n=5000, seed=0):
    """Paired MAE difference between two methods on the records both answered."""
    rng = np.random.default_rng(seed)
    p = d[[m, best, "gt"]].dropna()
    if len(p) < 20:
        return None
    em = (p[m] - p["gt"]).values
    e0 = (p[best] - p["gt"]).values
    diffs = np.empty(n)
    for i in range(n):
        j = rng.integers(0, len(p), len(p))
        diffs[i] = np.abs(em[j]).mean() - np.abs(e0[j]).mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return np.abs(em).mean() - np.abs(e0).mean(), lo, hi, len(p)
