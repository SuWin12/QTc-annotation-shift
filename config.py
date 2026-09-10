"""Paths, constants and device selection."""

from pathlib import Path

import numpy as np
import torch

QUICK   = False
RESTORE = False
EPOCHS  = 2 if QUICK else 40

BASE = Path.cwd()

LUDB_ROOT   = r"../../datasets/ludb/files/ludb/1.0.1/data"
ISP_ROOT    = r"../../datasets/ispdb"
PTB_ROOT    = r"../../datasets/ptbdb"
PTB_ANN     = f"{PTB_ROOT}/ptb_qt_annotations.csv"
PW_CKPT     = r"../../PhysioWave-main/pretrainedmodels/ecg.pth"
BEATSSL_DIR = r"../../beat_ssl/pretrained"
QTCNET_ROOT = r"../../qtcnet-main"

CKPT    = BASE / "checkpoints"
RES_DIR = BASE / "results"
FIG_DIR = BASE / "figures"
for _d in (CKPT, RES_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TARGET_FS = 500
WIN_SEC   = 10.0
LEN_5000  = int(TARGET_FS * WIN_SEC)
LEN_2048  = 2048
TRUST     = (500, 4500)
QT_MIN, QT_MAX, RR_MIN, RR_MAX = 200.0, 700.0, 0.30, 2.00

MODE   = "both"
CLAMP  = 20
THRESH = 450.0
SEED   = 42
RNG    = np.random.default_rng(0)

LABELS = {0: "none", 1: "P", 2: "QRS", 3: "T"}
STD12  = ["i", "ii", "iii", "avr", "avl", "avf",
          "v1", "v2", "v3", "v4", "v5", "v6"]
WAVE_COLOR = {1: "#e8b64c", 2: "#c1121f", 3: "#2a9d8f"}
WAVE_NAME  = {1: "P wave", 2: "QRS complex", 3: "T wave"}

PW_ARGS = dict(in_channels=12, max_level=3, wave_kernel_size=24,
               wavelet_names=['db4', 'db6', 'sym4', 'coif2'],
               use_separate_channel=True,
               patch_size=(1, 64), embed_dim=384, depth=8, num_heads=12,
               mlp_ratio=4.0, dropout=0.1, use_pos_embed=True,
               pos_embed_type='2d',
               task_type='classification', num_classes=5,
               head_config={'hidden_dims': [1024], 'dropout': 0.1,
                            'pooling': 'mean'})

DEV = "cuda" if torch.cuda.is_available() else \
      ("mps" if torch.backends.mps.is_available() else "cpu")
