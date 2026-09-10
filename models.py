"""The four method families, their predictors, and the training loop."""

import glob
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .config import (BEATSSL_DIR, CKPT, LEN_2048, LEN_5000, PW_ARGS, PW_CKPT,
                     QTCNET_ROOT, TRUST)
from .data import prep_signal, runs_of

sys.path.append(QTCNET_ROOT)
sys.path.append(r"../../PhysioWave-main")
sys.path.append(r"../../beat_ssl")

from inception_model import ECGModel
from model import BERTWaveletTransformer
from src.models.backbone.resnet import resnet1d18
from src.models.decoder import ResNet1DDecoder

import neurokit2 as nk


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def pick_device(override=None):
    if override:
        return override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class UNet1D(nn.Module):
    """[B, 1, 5000] -> [B, 4, 5000]. Trained from scratch on ISP."""

    def __init__(self, ch=16, n_cls=4):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(
                nn.Conv1d(i, o, 7, padding=3), nn.BatchNorm1d(o), nn.ReLU(),
                nn.Conv1d(o, o, 7, padding=3), nn.BatchNorm1d(o), nn.ReLU())

        self.e1, self.e2, self.e3 = blk(1, ch), blk(ch, ch * 2), blk(ch * 2, ch * 4)
        self.pool = nn.MaxPool1d(2)
        self.b = blk(ch * 4, ch * 8)
        self.u3, self.d3 = nn.ConvTranspose1d(ch * 8, ch * 4, 2, 2), blk(ch * 8, ch * 4)
        self.u2, self.d2 = nn.ConvTranspose1d(ch * 4, ch * 2, 2, 2), blk(ch * 4, ch * 2)
        self.u1, self.d1 = nn.ConvTranspose1d(ch * 2, ch, 2, 2), blk(ch * 2, ch)
        self.out = nn.Conv1d(ch, n_cls, 1)

    def forward(self, x):
        L = x.shape[-1]
        pad = (8 - L % 8) % 8
        if pad:
            x = F.pad(x, (0, pad))
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b  = self.b(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(b),  e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)[:, :, :L]


class BeatSSLSeg(nn.Module):
    """[B, 12, 5000] -> [B, 4, 5000]. Frozen pretrained encoder, decoder on ISP."""

    def __init__(self, pre_enc_dir=None, n_cls=4, freeze_encoder=True, verbose=False):
        super().__init__()
        self.encoder = resnet1d18(num_classes=128, input_channels=12)
        self.decoder = ResNet1DDecoder(output_channels=n_cls)
        self.freeze_encoder = freeze_encoder
        d = pre_enc_dir or BEATSSL_DIR
        sd = torch.load(Path(d).resolve() / "ecg_encoder_model.pth",
                        map_location="cpu", weights_only=False)
        for k in ("state_dict", "model_state_dict", "model"):
            if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
        enc = {k[len("encoder."):]: v for k, v in sd.items()
               if k.startswith("encoder.")} or dict(sd)
        before = {k: v.clone() for k, v in self.encoder.state_dict().items()}
        self.encoder.load_state_dict(enc, strict=False)
        changed = sum(1 for k in before
                      if not torch.equal(before[k], self.encoder.state_dict()[k]))
        assert changed > 0, "BeatSSL encoder did not load"
        if verbose:
            print(f"  BeatSSL: {changed}/{len(before)} tensors loaded")
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                _, e = self.encoder(x)
        else:
            _, e = self.encoder(x)
        return self.decoder(e)[:, :, :LEN_5000]


class PhysioWaveSeg(nn.Module):
    """[B, 12, 2048] -> [B, 4, 2048]. Pools frequency, keeps the time axis."""

    def __init__(self, ckpt=None, n_cls=4, freeze_encoder=True, verbose=False):
        super().__init__()
        self.encoder = BERTWaveletTransformer(**PW_ARGS)
        ck = torch.load(ckpt or PW_CKPT, map_location="cpu", weights_only=False)
        sd = ck["model_state_dict"] if "model_state_dict" in ck else ck

        miss, _ = self.encoder.load_state_dict(sd, strict=False)
        enc_miss = [k for k in miss if not k.startswith(("task_heads", "head"))]
        assert not enc_miss, f"PhysioWave weights missing: {enc_miss[:5]}"
        self.fp, self.tp, self.D = 48, 32, 384
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        def up(i, o):
            return nn.Sequential(
                nn.ConvTranspose1d(i, o, 4, stride=2, padding=1),
                nn.BatchNorm1d(o), nn.GELU())

        self.dec = nn.Sequential(up(384, 256), up(256, 192), up(192, 128),
                                 up(128, 96), up(96, 64), up(64, 32),
                                 nn.Conv1d(32, n_cls, 3, padding=1))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                f = self.encoder.forward_features(x)
        else:
            f = self.encoder.forward_features(x)
        B, N, D = f.shape
        f = f.view(B, self.fp, self.tp, D).mean(1).permute(0, 2, 1)
        return self.dec(f)[:, :, :LEN_2048]


QTCNET_LEADS = ["i", "ii", "iii", "avr", "avf", "avl",
                "v1", "v2", "v3", "v4", "v5", "v6"]
_QTCNET = None


def load_qtcnet():
    """Released QTcNet weights, built on CPU so any device can host them."""
    global _QTCNET
    if _QTCNET is None:
        hits = (glob.glob(f"{QTCNET_ROOT}/**/best_model.pth.tar", recursive=True)
                or glob.glob("../qtcnet-main/**/best_model.pth.tar", recursive=True)
                or glob.glob("**/best_model.pth.tar", recursive=True))
        if not hits:
            raise FileNotFoundError("QTcNet checkpoint not found")
        w = hits[0]
        m = ECGModel(c_in=12, c_out=1)
        ck = torch.load(w, map_location="cpu")
        for k in ("state_dict", "model_state_dict", "model", "net"):
            if isinstance(ck, dict) and k in ck:
                ck = ck[k]
                break
        ck = {("model." + k if not k.startswith("model.") else k): v
              for k, v in ck.items()}
        m.load_state_dict(ck, strict=True)
        m.eval()
        _QTCNET = m
        print(f"QTcNet loaded from {w}")
    return _QTCNET


def predict_qtcnet(rec):
    """Corrected interval predicted directly, no boundaries."""
    idx = [rec["leads"].index(l) for l in QTCNET_LEADS]
    s = np.nan_to_num(np.asarray(rec["sig12"], float)[:, idx])
    if rec["fs"] != 100:
        s = scipy.signal.resample(s, int(len(s) * 100 / rec["fs"]), axis=0)
    if s.shape[0] < 1000:
        s = np.pad(s, ((0, 1000 - s.shape[0]), (0, 0)))
    with torch.no_grad():
        x = torch.tensor(s[:1000].T, dtype=torch.float32).unsqueeze(0)
        return float(load_qtcnet()(x).squeeze())


def predict_wavelet(rec, method="dwt"):
    """NeuroKit dwt or cwt -> (onsets, offsets, rpeaks)."""
    fs  = rec["fs"]
    sig = np.asarray(rec["sig"], float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = nk.ecg_clean(sig, sampling_rate=fs)
        _, info = nk.ecg_peaks(clean, sampling_rate=fs)
        rp = info["ECG_R_Peaks"]
        if len(rp) < 4:
            return np.array([]), np.array([]), np.asarray(rp, float)
        _, w = nk.ecg_delineate(clean, rp, sampling_rate=fs, method=method)

    on  = np.sort(np.asarray(w["ECG_R_Onsets"], float)[
                  np.isfinite(np.asarray(w["ECG_R_Onsets"], float))])
    off = np.sort(np.asarray(w["ECG_T_Offsets"], float)[
                  np.isfinite(np.asarray(w["ECG_T_Offsets"], float))])

    ons, offs = [], []
    for o in on:
        cand = off[(off > o) & (off - o < 0.7 * fs)]
        if len(cand):
            ons.append(o)
            offs.append(cand[0])
    return np.asarray(ons, float), np.asarray(offs, float), np.asarray(rp, float)


def mask_to_fiducials(lab, fs_eff, trust):
    """QRS onset paired with the next T offset within 0.7 s."""
    lo, hi = trust
    qrs, t = runs_of(lab, 2, lo, hi), runs_of(lab, 3, lo, hi)
    ons, offs = [], []
    for qon, _ in qrs:
        cand = [b for a, b in t if a > qon and (b - qon) < 0.7 * fs_eff]
        if cand:
            ons.append(qon)
            offs.append(cand[0])
    return np.asarray(ons, float), np.asarray(offs, float)


def _predict_mask(rec, model, target, leads):
    dev = next(model.parameters()).device
    x = torch.tensor(prep_signal(rec, target, leads)).unsqueeze(0).to(dev)
    model.eval()
    with torch.no_grad():
        return model(x).argmax(1)[0].cpu().numpy()


def predict_unet(rec, model):
    lab = _predict_mask(rec, model, LEN_5000, 1)
    fs_eff = LEN_5000 / (len(rec["sig"]) / rec["fs"])
    on, off = mask_to_fiducials(lab, fs_eff, TRUST)
    s = len(rec["sig"]) / LEN_5000
    return on * s, off * s, lab


def predict_beatssl(rec, model):
    lab = _predict_mask(rec, model, LEN_5000, 12)
    fs_eff = LEN_5000 / (len(rec["sig"]) / rec["fs"])
    on, off = mask_to_fiducials(lab, fs_eff, TRUST)
    s = len(rec["sig"]) / LEN_5000
    return on * s, off * s, lab


def predict_physiowave(rec, model):
    lab = _predict_mask(rec, model, LEN_2048, 12)
    fs_eff = LEN_2048 / (len(rec["sig"]) / rec["fs"])
    tr = (int(TRUST[0] * LEN_2048 / LEN_5000), int(TRUST[1] * LEN_2048 / LEN_5000))
    on, off = mask_to_fiducials(lab, fs_eff, tr)
    s = len(rec["sig"]) / LEN_2048
    return on * s, off * s, lab


SEG_PRED = {"unet": predict_unet, "beatssl": predict_beatssl,
            "physiowave": predict_physiowave}


def train_seg(build, train_ds, val_ds, *, name, seed=42, epochs=40, lr=1e-3,
              batch=16, patience=10, device=None, verbose=True):
    """Cross-entropy over four classes, keeping the best-validation-F1 epoch."""
    set_seed(seed)
    dev = pick_device(device)
    model = build().to(dev)

    with torch.no_grad():
        x0, _ = train_ds[0]
        model(x0.unsqueeze(0).to(dev))

    params = [p for p in model.parameters() if p.requires_grad]
    opt    = torch.optim.Adam(params, lr=lr)
    lossf  = nn.CrossEntropyLoss()
    tl = DataLoader(train_ds, batch_size=batch, shuffle=True)
    vl = DataLoader(val_ds,   batch_size=batch)

    hist = {"train_loss": [], "val_loss": [], "val_f1": []}
    best, best_ep, bad = -1.0, 0, 0
    path = CKPT / f"{name}_s{seed}.pt"
    print(f"{name} | seed {seed} | {dev} | lr {lr:.2g} | batch {batch} | "
          f"{sum(p.numel() for p in params) / 1e6:.2f} M trainable | "
          f"train {len(train_ds)} val {len(val_ds)}")

    for ep in range(1, epochs + 1):
        model.train()
        tot = n = 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out = model(x)
            L = min(out.shape[-1], y.shape[-1])
            loss = lossf(out[:, :, :L], y[:, :L])
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
            n += len(x)
        tr_loss = tot / max(n, 1)

        model.eval()
        vt = vn = 0
        P, T = [], []
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(dev), y.to(dev)
                out = model(x)
                L = min(out.shape[-1], y.shape[-1])
                vt += lossf(out[:, :, :L], y[:, :L]).item() * len(x)
                vn += len(x)
                P.append(out[:, :, :L].argmax(1).cpu().numpy().ravel())
                T.append(y[:, :L].cpu().numpy().ravel())
        va_loss = vt / max(vn, 1)
        f1 = f1_score(np.concatenate(T), np.concatenate(P),
                      average="macro", labels=[1, 2, 3], zero_division=0)

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["val_f1"].append(f1)

        if f1 > best:
            best, best_ep, bad = f1, ep, 0
            torch.save(model.state_dict(), path)
        else:
            bad += 1
        if verbose and (ep == 1 or ep % 5 == 0):
            print(f"  ep {ep:3d}: train {tr_loss:.4f} | val {va_loss:.4f} | F1 {f1:.4f}")
        if bad >= patience:
            print(f"  early stop at ep {ep} (no gain for {patience})")
            break

    model.load_state_dict(torch.load(path, map_location=dev))
    print(f"  -> best F1 {best:.4f} @ epoch {best_ep}\n")
    return model, hist


def restore_seg(specs, aliases=None):
    """Load saved checkpoints on CPU, then move each to the device it was run on."""
    aliases = aliases or {
        "unet":       ["sweep_unet_lr0.001_s42.pt", "seg_unet_s42.pt"],
        "beatssl":    ["sweep_beatssl_lr0.001_s42.pt", "seg_beatssl_s42.pt"],
        "physiowave": ["sweep_physiowave_lr0.001_s42.pt", "seg_pw_s42.pt"]}
    seg = {}
    for name, build, _ds, _tag, _batch, dev in specs:
        path = next((CKPT / f for f in aliases[name] if (CKPT / f).exists()), None)
        if path is None:
            print(f"  MISSING {name}: no checkpoint found")
            continue
        m = build()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        seg[name] = m.to(dev).eval()
        print(f"  {name:<11} restored from {path.name} on {dev}")
    return seg
