"""Stage one: the geometric T-offset construction, under a clamp."""

import numpy as np
from scipy.signal import savgol_filter


def _baseline(sig, t_off_init, fs, lo_ms=60, hi_ms=160):
    a = int(t_off_init + lo_ms * fs / 1000)
    b = int(t_off_init + hi_ms * fs / 1000)
    a, b = max(0, min(a, len(sig) - 1)), max(1, min(b, len(sig)))
    return float(np.median(sig)) if b - a < 3 else float(np.median(sig[a:b]))


def _t_peak(sig, qrs_off, t_off_init, base):
    a, b = int(qrs_off), int(t_off_init)
    if b - a < 5:
        return None, 0
    seg = sig[a:b] - base
    k = int(np.argmax(np.abs(seg)))
    return a + k, (1 if seg[k] >= 0 else -1)


def _smooth(x, fs):
    w = int(round(0.020 * fs)) | 1
    w = max(5, min(w, len(x) - 1 if len(x) % 2 == 0 else len(x) - 2))
    return savgol_filter(x, w, 3) if len(x) > w else np.asarray(x, float)


def t_end_tangent(sig, t_peak, base, pol, fs, search_ms=120):
    """Steepest point on the falling limb, tangent extended to baseline."""
    y = pol * (np.asarray(sig, float) - base)
    a = int(t_peak) + 1
    b = min(len(y) - 1, int(t_peak + search_ms * fs / 1000))
    if b - a < 5:
        return None
    ys = _smooth(y[a:b + 1], fs)
    d = np.gradient(ys)
    k = int(np.argmin(d))
    if d[k] >= -1e-9:
        return None
    return float(a + k - ys[k] / d[k])


def t_end_knee(sig, t_peak, base, pol, fs, search_ms=200):
    """Point furthest from the chord joining T peak to end of search window."""
    y = pol * (np.asarray(sig, float) - base)
    a = int(t_peak)
    b = min(len(y) - 1, int(t_peak + search_ms * fs / 1000))
    if b - a < 8:
        return None
    ys = _smooth(y[a:b + 1], fs)
    x = np.arange(len(ys), dtype=float)
    dx, dy = x[-1] - x[0], ys[-1] - ys[0]
    den = np.hypot(dx, dy)
    if den < 1e-9:
        return None
    dist = ((x - x[0]) * dy - (ys - ys[0]) * dx) / den
    return float(a + int(np.argmax(dist)))


def refine_t_offset(sig, qrs_off, t_off_init, fs, max_shift_ms=30, mode="both"):
    """Construct the T offset, clamped to max_shift_ms from the network mark."""
    sig = np.asarray(sig, float)
    t0 = int(np.clip(t_off_init, 1, len(sig) - 2))
    base = _baseline(sig, t0, fs)
    pk, pol = _t_peak(sig, qrs_off, t0, base)
    if pk is None:
        return float(t0)
    cands = []
    if mode in ("tangent", "both"):
        c = t_end_tangent(sig, pk, base, pol, fs)
        if c is not None:
            cands.append(c)
    if mode in ("knee", "both"):
        c = t_end_knee(sig, pk, base, pol, fs)
        if c is not None:
            cands.append(c)
    if not cands:
        return float(t0)
    lim = max_shift_ms * fs / 1000.0
    return float(np.clip(float(np.mean(cands)), t0 - lim, t0 + lim))


def apply_refinement(sig, ons, offs, fs, mode="both", max_shift_ms=30):
    """Refine every T offset in one record."""
    if len(offs) == 0:
        return ons, offs
    qrs_dur = int(round(0.10 * fs))
    out = []
    for k, t_off in enumerate(offs):
        r = refine_t_offset(sig, float(ons[k]) + qrs_dur, float(t_off),
                            fs, max_shift_ms, mode)
        out.append(r if r is not None and np.isfinite(r) else t_off)
    return ons, np.asarray(out, float)
