"""Tests of the annotation offset against heart rate and correction formula."""

import numpy as np
import pandas as pd
from scipy import stats

from .config import CLAMP, MODE
from .evaluate import gt_fiducials, qtc, qtc_errors, r_peaks

FORMS = ("bazett", "fridericia", "framingham", "hodges", "rautaharju")


def reference_rows(recs, tag):
    """Reference QT, RR and QTc per record, from the human marks alone."""
    out = []
    for rec in recs:
        try:
            g_on, g_off = gt_fiducials(rec)
            rp = r_peaks(rec)
        except Exception:
            continue
        if len(g_on) == 0 or len(rp) < 2:
            continue
        fs = rec["fs"]
        rr = float(np.median(np.diff(rp)) / fs)
        if not np.isfinite(rr) or rr <= 0:
            continue
        qt = (np.asarray(g_off, float) - np.asarray(g_on, float)) / fs
        qt = qt[(qt > 0.20) & (qt < 0.70)]
        if len(qt) == 0:
            continue
        row = {"db": tag, "id": rec["id"], "QT": 1000.0 * float(np.median(qt)),
               "RR": rr, "HR": 60.0 / rr}
        for f in FORMS:
            row[f] = qtc(g_on, g_off, rp, fs, f)
        out.append(row)
    return out


def offset_confound(isp, ludb, verbose=True):
    """Raw QT gap, heart-rate distributions, the gap by formula, and by HR bin."""
    D = pd.DataFrame(reference_rows(isp, "ISP") + reference_rows(ludb, "LUDB"))
    D = D[np.isfinite(D["QT"]) & np.isfinite(D["RR"])].reset_index(drop=True)
    A, B = D[D.db == "ISP"], D[D.db == "LUDB"]

    qt_gap = A["QT"].mean() - B["QT"].mean()
    _, p_qt = stats.ttest_ind(A["QT"], B["QT"], equal_var=False)
    ks, p_ks = stats.ks_2samp(A["HR"], B["HR"])

    T3 = pd.DataFrame([{"formula": f, "ISP": A[f].dropna().mean(),
                        "LUDB": B[f].dropna().mean(),
                        "gap": A[f].dropna().mean() - B[f].dropna().mean()}
                       for f in FORMS]).round(2)

    edges = np.percentile(D["HR"], [0, 20, 40, 60, 80, 100])
    edges[-1] += 1e-6
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        a = A[(A.HR >= lo) & (A.HR < hi)]
        b = B[(B.HR >= lo) & (B.HR < hi)]
        if len(a) < 10 or len(b) < 10:
            continue
        rows.append({"HR bin": f"{lo:.0f}-{hi:.0f}", "n ISP": len(a),
                     "n LUDB": len(b),
                     "QT gap": a["QT"].mean() - b["QT"].mean(),
                     "QTc gap": a["bazett"].mean() - b["bazett"].mean()})
    T4 = pd.DataFrame(rows).round(2)

    pooled = {}
    if len(T4):
        w = T4["n ISP"] + T4["n LUDB"]
        pooled = {"QT": float((T4["QT gap"] * w).sum() / w.sum()),
                  "QTc": float((T4["QTc gap"] * w).sum() / w.sum())}

    if verbose:
        print(f"ISP {len(A)} records, LUDB {len(B)} records\n")
        print(f"raw QT      ISP {A['QT'].mean():7.2f}  LUDB {B['QT'].mean():7.2f}"
              f"  gap {qt_gap:+.2f} ms   p = {p_qt:.2e}")
        print(f"heart rate  ISP {A['HR'].mean():7.2f}  LUDB {B['HR'].mean():7.2f}"
              f"  KS {ks:.3f}   p = {p_ks:.2e}\n")
        print(T3.to_string(index=False))
        print(f"\nspread of the gap across formulae: "
              f"{T3['gap'].max() - T3['gap'].min():.2f} ms\n")
        if len(T4):
            print(T4.to_string(index=False))
            print(f"\npooled within-bin QT  gap {pooled['QT']:+.2f} ms")
            print(f"pooled within-bin QTc gap {pooled['QTc']:+.2f} ms")

    return {"records": D, "by_formula": T3, "by_hr_bin": T4,
            "qt_gap": qt_gap, "p_qt": p_qt, "ks": ks, "p_ks": p_ks,
            "pooled": pooled}


def formula_robustness(models, isp_test, ludb, forms=("bazett", "fridericia",
                                                      "hodges"),
                       mode=MODE, clamp=CLAMP, verbose=True):
    """Refinement gain, harm reversal and constant-versus-geometry, by formula."""
    ms = [m for m in ("unet", "beatssl", "physiowave") if m in models]

    rows = []
    for f in forms:
        for m in ms:
            P0, R0 = qtc_errors(ludb, m, models, formula=f)
            P1, R1 = qtc_errors(ludb, m, models, refine=mode, clamp=clamp, formula=f)
            if not len(P0) or not len(P1):
                continue
            e0, e1 = P0 - R0, P1 - R1
            e0, e1 = e0[np.isfinite(e0)], e1[np.isfinite(e1)]
            rows.append({"formula": f, "model": m,
                         "bias none": e0.mean(), "MAE none": np.abs(e0).mean(),
                         "bias ref": e1.mean(), "MAE ref": np.abs(e1).mean(),
                         "gain": np.abs(e0).mean() - np.abs(e1).mean()})
    GAIN = pd.DataFrame(rows).round(2)

    rows = []
    for f in forms:
        for m in ms:
            for tag, recs in (("ISP held out", isp_test), ("LUDB unseen", ludb)):
                P0, R0 = qtc_errors(recs, m, models, formula=f)
                P1, R1 = qtc_errors(recs, m, models, refine=mode, clamp=clamp,
                                    formula=f)
                if not len(P0) or not len(P1):
                    continue
                e0, e1 = P0 - R0, P1 - R1
                e0, e1 = e0[np.isfinite(e0)], e1[np.isfinite(e1)]
                rows.append({"formula": f, "model": m, "data": tag,
                             "without": np.abs(e0).mean(), "bias w/o": e0.mean(),
                             "with": np.abs(e1).mean(), "bias with": e1.mean(),
                             "change": np.abs(e1).mean() - np.abs(e0).mean()})
    REV = pd.DataFrame(rows).round(2)
    piv = REV.pivot_table(index=["formula", "model"], columns="data",
                          values="change").round(2)
    ok = tot = 0
    for (_f, _m), r in piv.iterrows():
        i_c, l_c = r.get("ISP held out", np.nan), r.get("LUDB unseen", np.nan)
        if np.isfinite(i_c) and np.isfinite(l_c):
            tot += 1
            ok += int(i_c > 0 and l_c < 0)

    rows = []
    for f in forms:
        for m in ms:
            best, best_s = np.inf, None
            for s in np.arange(-40, 11, 2):
                P, R = qtc_errors(ludb, m, models, shift_ms=float(s), formula=f)
                if not len(P):
                    continue
                e = P - R
                e = e[np.isfinite(e)]
                if len(e) and np.abs(e).mean() < best:
                    best, best_s = np.abs(e).mean(), int(s)
            Pg, Rg = qtc_errors(ludb, m, models, refine=mode, clamp=clamp, formula=f)
            eg = Pg - Rg
            eg = eg[np.isfinite(eg)]
            rows.append({"formula": f, "model": m, "constant": best,
                         "geometry": np.abs(eg).mean(),
                         "gap": abs(best - np.abs(eg).mean()),
                         "fitted shift": best_s})
    CONST = pd.DataFrame(rows).round(2)

    if verbose:
        print(GAIN.to_string(index=False))
        print("\n" + piv.to_string())
        print(f"\nreversal holds in {ok} of {tot} formula and model combinations\n")
        print(CONST.to_string(index=False))

    return {"gain": GAIN, "reversal": REV, "reversal_pivot": piv,
            "reversal_count": (ok, tot), "constant": CONST}
