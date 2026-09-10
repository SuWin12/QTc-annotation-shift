"""Three training objectives: cross-entropy, boundary-weighted, interval-aware."""

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .config import CKPT, DEV, EPOCHS, SEED
from .data import SegDS12
from .models import BeatSSLSeg

P_CLS, QRS_CLS, T_CLS = 1, 2, 3
LAM = 0.005


def soft_edge(prob, anchor, half, rising, temp=0.05):
    """Sub-sample boundary position, differentiable in the class posteriors."""
    n = prob.shape[0]
    a, b = max(0, int(anchor) - half), min(n, int(anchor) + half + 1)
    if b - a < 3:
        return None
    seg = prob[a:b]
    d = seg[1:] - seg[:-1]
    d = d if rising else -d
    d = d / (d.max().detach() + 1e-6)
    w = torch.softmax(d / temp, dim=0)
    idx = torch.arange(a, b - 1, device=prob.device, dtype=prob.dtype) + 0.5
    return (w * idx).sum()


def true_beats(y, max_beats=99):
    """(QRS onset, T offset) pairs from one reference label sequence."""
    yn = y.detach().cpu().numpy()
    q = np.flatnonzero(np.diff((yn == QRS_CLS).astype(np.int8), prepend=0) == 1)
    t = np.flatnonzero(np.diff((yn == T_CLS).astype(np.int8), append=0) == -1)
    out = []
    for qq in q:
        nx = t[t > qq]
        if nx.size:
            out.append((int(qq), int(nx[0])))
        if len(out) >= max_beats:
            break
    return out


def pred_anchors(logits_i, max_beats=99):
    """The network's own boundaries, detached, used only to place windows."""
    lab = logits_i.argmax(0).detach().cpu().numpy()
    q = np.flatnonzero(np.diff((lab == QRS_CLS).astype(np.int8), prepend=0) == 1)
    t = np.flatnonzero(np.diff((lab == T_CLS).astype(np.int8), append=0) == -1)
    out = []
    for qq in q:
        nx = t[t > qq]
        if nx.size:
            out.append((int(qq), int(nx[0])))
        if len(out) >= max_beats:
            break
    return out


def interval_loss(logits, y, fs=500, half_ms=40, w_qt=1.0, w_pt=0.25, tol_ms=150):
    """QT error in milliseconds, differentiable, with both endpoint terms."""
    probs = torch.softmax(logits, dim=1)
    half = int(half_ms * fs / 1000)
    tol  = tol_ms * fs / 1000
    ms   = 1000.0 / fs
    tot, cnt = torch.zeros((), device=logits.device), 0
    for i in range(logits.shape[0]):
        truth = true_beats(y[i])
        if not truth:
            continue
        tq = np.array([a for a, _ in truth])
        tt = np.array([b for _, b in truth])
        pq, pt = probs[i, QRS_CLS], probs[i, T_CLS]
        for q_a, t_a in pred_anchors(logits[i]):
            j = int(np.argmin(np.abs(tq - q_a)))
            if abs(tq[j] - q_a) > tol or abs(tt[j] - t_a) > tol:
                continue
            q_h = soft_edge(pq, q_a, half, rising=True)
            t_h = soft_edge(pt, t_a, half, rising=False)
            if q_h is None or t_h is None:
                continue
            e_qt = ((t_h - q_h) - float(tt[j] - tq[j])) * ms
            e_q  = (q_h - float(tq[j])) * ms
            e_t  = (t_h - float(tt[j])) * ms
            z = torch.zeros_like(e_qt)
            tot = tot + w_qt * F.smooth_l1_loss(e_qt, z, beta=5.0) \
                      + w_pt * F.smooth_l1_loss(e_q,  z, beta=5.0) \
                      + w_pt * F.smooth_l1_loss(e_t,  z, beta=5.0)
            cnt += 1
    return tot / max(cnt, 1)


def boundary_weight_map(y, fs=500, width_ms=40, peak=5.0):
    """Objective B: weight rises near every class transition."""
    w = torch.ones_like(y, dtype=torch.float32)
    half = max(1, int(width_ms * fs / 1000))
    changed = (y[:, 1:] != y[:, :-1]).float()
    changed = F.pad(changed.unsqueeze(1), (1, 0))
    k = torch.ones(1, 1, 2 * half + 1, device=y.device)
    near = F.conv1d(changed, k, padding=half).squeeze(1).clamp(0, 1)
    return w + (peak - 1.0) * near


def train_obj(mode, name, isp_train, isp_val, epochs=None, batch=4, lr=1e-3,
              dev=None):
    """mode: 'ce' | 'bw' | 'interval'. Everything but the loss is identical."""
    epochs = epochs or EPOCHS
    dev = dev or DEV
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = BeatSSLSeg().to(dev)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    dl_tr = torch.utils.data.DataLoader(SegDS12(isp_train), batch_size=batch,
                                        shuffle=True)
    dl_va = torch.utils.data.DataLoader(SegDS12(isp_val), batch_size=batch)
    hist = {"ce": [], "iv": [], "val_f1": []}
    best, best_sd = -1.0, None

    for ep in range(epochs):
        model.train()
        run_ce = run_iv = 0.0
        nb = 0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev).long()
            logits = model(x)
            if mode == "bw":
                w  = boundary_weight_map(y, fs=500)
                ce = (F.cross_entropy(logits, y, reduction="none") * w).mean()
                iv = torch.zeros((), device=dev)
            else:
                ce = F.cross_entropy(logits, y)
                iv = interval_loss(logits, y) if mode == "interval" \
                     else torch.zeros((), device=dev)
            loss = ce + LAM * iv
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run_ce += ce.item()
            run_iv += float(iv)
            nb += 1

        model.eval()
        inter = np.zeros(4)
        union = np.zeros(4)
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(dev), y.to(dev).long()
                pr = model(x).argmax(1)
                for c in range(4):
                    inter[c] += ((pr == c) & (y == c)).sum().item()
                    union[c] += ((pr == c).sum() + (y == c).sum()).item()
        f1 = float(np.mean(2 * inter[1:] / np.maximum(union[1:], 1)))
        hist["ce"].append(run_ce / nb)
        hist["iv"].append(run_iv / nb)
        hist["val_f1"].append(f1)
        if f1 > best:
            best, best_sd = f1, copy.deepcopy(model.state_dict())
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  {name} ep{ep:02d}  ce {run_ce / nb:.4f}  "
                  f"interval {run_iv / nb:7.2f} ms  valF1 {f1:.4f}")

    model.load_state_dict(best_sd)
    print(f"  -> best val F1 {best:.4f}")
    torch.save(model.state_dict(), CKPT / f"obj_{mode}_s{SEED}.pt")
    return model.eval(), hist, best
