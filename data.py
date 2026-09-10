"""Record loading, PTB reader agreement, and the segmentation datasets."""

import ast
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal
import torch
import wfdb
from torch.utils.data import Dataset

from .config import (ISP_ROOT, LEN_2048, LEN_5000, LUDB_ROOT, PTB_ANN,
                     PTB_ROOT)


def load_ludb(root=LUDB_ROOT, lead="ii"):
    out, skipped = [], []
    #get hea files
    for hea in sorted(glob.glob(f"{root}/*.hea")):
        p = hea[:-4]  #remove ".hea" name

        try:
            rec = wfdb.rdrecord(p)

            idx = [s.lower() for s in rec.sig_name].index(lead)  #find lead index
            sig = rec.p_signal[:, idx].astype(np.float32)
            ann = wfdb.rdann(p, extension=lead)  #raw

            #mask
            samples = ann.sample
            mask = np.zeros(len(sig), int)
            y = list(ann.symbol)
            i = 0
            while i < len(y):
                if y[i] == "(" and i + 2 < len(y) and y[i + 2] == ")":
                    cls = {"p": 1, "N": 2, "t": 3}.get(y[i + 1])
                    if cls:
                        mask[samples[i]:samples[i + 2] + 1] = cls
                    i += 3
                else:
                    i += 1

            out.append({
                "id": f"ludb_{Path(p).name}",
                "path": p,
                "fs": int(rec.fs),
                "sig": sig,
                "sig12": rec.p_signal.astype(np.float32),
                "mask": mask,
                "ann": ann,  #raw wfdb triples
                "leads": [s.lower() for s in rec.sig_name],
            })

        except Exception as e:
            skipped.append(f"{p} has problem as {e}")

    print(f"skipped: {len(skipped)}")
    print(f"loaded: {len(out)}")
    return out


ISPMask = {0: 1, 1: 2, 2: 3}


def load_isp(root=ISP_ROOT, which="test", lead="ii"):
    csv = pd.read_csv(f"{root}/{which}_isp_delineation_data.csv")
    ann_col = next((c for c in csv.columns
                    if str(csv[c].iloc[0]).strip().startswith("[(")),
                   csv.columns[-1])
    out, skipped = [], []

    for _, r in csv.iterrows():
        rid = str(r["file_name"])
        p = f"{root}/{which}_data/{rid}"
        try:
            rec = wfdb.rdrecord(p)
            idx = [s.lower() for s in rec.sig_name].index(lead)
            sig = rec.p_signal.astype(np.float32)
            if np.nanmax(np.abs(sig)) > 50:
                sig = sig / 1000.0
            waves = ast.literal_eval(r[ann_col])

            #mask
            mask = np.zeros(len(sig), int)
            for cls, on, off in waves:
                m = ISPMask.get(cls)
                if m:
                    mask[on:min(off + 1, len(sig))] = m

            out.append({
                "id": f"ispdb_{Path(p).name}",
                "path": p,
                "fs": int(rec.fs),
                "sig": rec.p_signal[:, idx].astype(np.float32),
                "sig12": sig,
                "mask": mask,
                "waves": waves,  #diff from ludb; raw(class,on,off)tuples
                "leads": [s.lower() for s in rec.sig_name],
            })

        except Exception as e:
            skipped.append(f"{p} has problem as {e}")

    print(f"skipped: {len(skipped)}")
    print(f"loaded: {len(out)}")
    for s in skipped:
        print("       ", s)
    return out


COLS  = ["patient", "record", "q1", "q2", "q3", "q4", "q5", "q_med",
         "t1", "t2", "t3", "t4", "t5", "t_med"]
QCOLS = ["q1", "q2", "q3", "q4", "q5"]
TCOLS = ["t1", "t2", "t3", "t4", "t5"]
NUM   = COLS[2:]


def ptb_annotations():
    """The five reader marks per record, from the CSV alone."""
    csv = pd.read_csv(PTB_ANN, header=None, dtype=str)
    csv.columns = COLS
    for c in NUM:
        csv[c] = pd.to_numeric(
            csv[c].astype(str).str.replace(r"[^\d.\-]", "", regex=True),
            errors="coerce")
    ann = csv[csv[NUM].notna().sum(axis=1) >= 10].copy()
    ann["record"] = ann["record"].astype(str).str.strip()
    return ann


def ptb_reader_agreement(verbose=True):
    """Median standard deviation across five readers, in milliseconds."""
    ann = ptb_annotations()
    q  = ann[QCOLS].to_numpy(float)
    t  = ann[TCOLS].to_numpy(float)
    qt = t - q
    sd_q, sd_t, sd_qt = (np.nanstd(x, axis=1, ddof=1) for x in (q, t, qt))
    out = {"n": len(ann),
           "qrs_onset_sd": float(np.nanmedian(sd_q)),
           "t_offset_sd":  float(np.nanmedian(sd_t)),
           "qt_sd":        float(np.nanmedian(sd_qt))}
    if verbose:
        print(f"  records with five reader marks: {out['n']}")
        print(f"  QRS onset   median SD {out['qrs_onset_sd']:5.2f} ms")
        print(f"  T offset    median SD {out['t_offset_sd']:5.2f} ms")
        print(f"  QT interval median SD {out['qt_sd']:5.2f} ms")
    return out


def load_ptb(root=PTB_ROOT, lead="ii", verbose=True):
    """PTB records with the median reader marks attached."""
    ann = ptb_annotations()

    lookup = {}
    for h in sorted(glob.glob(f"{root}/**/*.hea", recursive=True)):
        lookup[Path(h).stem] = h[:-4]
    if verbose:
        print(f"  header files found: {len(lookup)}")

    norm = lambda s: str(s).strip().lower().replace("_", "").replace("-", "")
    lookup_n = {norm(k): v for k, v in lookup.items()}
    ann["rec_path"] = ann["record"].map(lambda s: lookup_n.get(norm(s)))
    matched = ann.dropna(subset=["rec_path"]).reset_index(drop=True)
    if verbose:
        print(f"  annotation rows matched to a file: {len(matched)} of {len(ann)}")

    out, skipped = [], []
    for _, r in matched.iterrows():
        pth = str(r["rec_path"])
        try:
            rec = wfdb.rdrecord(pth)
            names = [s.lower() for s in rec.sig_name]
            if lead not in names:
                skipped.append(f"{pth}: no lead {lead}")
                continue
            sig = np.asarray(rec.p_signal[:, names.index(lead)], dtype=np.float32)

            q = np.array([r[c] for c in QCOLS], dtype=float)
            t = np.array([r[c] for c in TCOLS], dtype=float)
            ok = np.isfinite(q) & np.isfinite(t) & (t > q)
            if ok.sum() < 2:
                skipped.append(f"{pth}: fewer than two usable reader marks")
                continue

            out.append({
                "id":    f"ptbdb_{Path(pth).name}",
                "path":  pth,
                "fs":    int(rec.fs),
                "sig":   sig,
                "sig12": np.asarray(rec.p_signal, dtype=np.float32),
                "q_ref": float(np.median(q[ok])),
                "t_ref": float(np.median(t[ok])),
                "sd_qt": float(np.nanstd((t - q)[ok], ddof=1)),
                "leads": names,
            })
        except Exception as e:
            skipped.append(f"{pth}: {type(e).__name__}: {e}")

    if verbose:
        print(f"  loaded {len(out)}, skipped {len(skipped)}")
    return out


def runs_of(mask, c, lo=0, hi=None):
    """Contiguous runs of class c inside [lo, hi), at least three samples."""
    hi = hi or len(mask)
    seg = np.zeros(len(mask), int)
    seg[lo:hi] = (np.asarray(mask)[lo:hi] == c)
    e = np.diff(np.concatenate([[0], seg, [0]]))
    return [(s, t - 1) for s, t in zip(np.where(e == 1)[0], np.where(e == -1)[0])
            if t - 1 - s >= 2]


def prep_signal(rec, target, leads):
    """Resample to target length, z-score per lead, return (C, target)."""
    s = np.nan_to_num(np.asarray(rec["sig12"], np.float32))[:, :12]
    if s.shape[0] != target:
        s = scipy.signal.resample(s, target, axis=0)
    s = (s - s.mean(0)) / (s.std(0) + 1e-6)
    s = s.T
    return s.astype(np.float32) if leads == 12 else s[1:2].astype(np.float32)


def prep_mask(rec, target):
    m = np.asarray(rec["mask"], int)
    if len(m) != target:
        m = m[np.linspace(0, len(m) - 1, target).round().astype(int)]
    return m


class SegDS(Dataset):
    """Per-sample four-class labels for one set of records."""

    def __init__(self, recs, leads=12, target=LEN_5000):
        self.items = [(torch.tensor(prep_signal(r, target, leads)),
                       torch.tensor(prep_mask(r, target), dtype=torch.long))
                      for r in recs]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


SegDS1L = lambda recs: SegDS(recs,  1, LEN_5000)
SegDS12 = lambda recs: SegDS(recs, 12, LEN_5000)
SegDSPW = lambda recs: SegDS(recs, 12, LEN_2048)


def split_by_record(recs, seed=42, frac_train=0.70, frac_val=0.10):
    """Disjoint train / validation / held-out partitions, split by record id."""
    ids = sorted({r["id"] for r in recs})
    rs = np.random.RandomState(seed)
    rs.shuffle(ids)
    n_tr, n_va = int(frac_train * len(ids)), int(frac_val * len(ids))
    tr, va = set(ids[:n_tr]), set(ids[n_tr:n_tr + n_va])
    te = set(ids[n_tr + n_va:])
    assert not (tr & te) and not (tr & va) and not (va & te)
    return ([r for r in recs if r["id"] in tr],
            [r for r in recs if r["id"] in va],
            [r for r in recs if r["id"] in te])
