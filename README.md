# Correcting Annotation Shift in QTc Estimation from the Electrocardiogram

Six methods from four families are compared for
corrected QT interval estimation, trained on one annotated database and tested
on a second that is withheld until evaluation. Two constructions are added on
top: a differentiable interval term in the training objective, and a two-stage
post-processing pipeline of a clamped geometric T-offset correction followed by
a label-free rejection gate.


## Data

Three databases, none redistributed here.

| Database | Records | Rate | Role |
|---|---|---|---|
| ISP | 475 (462 usable) | 1000 Hz | training, validation, held out |
| LUDB | 200 | 500 Hz | test, unseen until evaluation |
| PTB | 549 | 1000 Hz | five-reader agreement only |

Set the paths in `config.py`. The pretrained encoders and the released
QTcNet weights are expected at `PW_CKPT`, `BEATSSL_DIR` and `QTCNET_ROOT`.

## Running

```bash
pip install -r requirements.txt
jupyter lab qtc_main.ipynb
```

Run the cells in order. `RESTORE = True` in `config.py` loads saved
checkpoints and skips training. Training three segmentation models plus three
objectives takes a few hours on an Apple M-series GPU; everything after that is
minutes on CPU.

PhysioWave is built and run on CPU because its wavelet front end has no
Metal kernel. All checkpoints are loaded with `map_location="cpu"` and moved
afterwards, so weights trained on CUDA restore on MPS without change.


## Methods

| Method | Family | Trainable | Input |
|---|---|---|---|
| DWT, CWT | rule-based delineation | none | lead II |
| U-Net 1D | supervised segmentation | 0.36 M | lead II, 5000 samples |
| BeatSSL | frozen pretrained encoder, trained decoder | 0.70 M | 12 leads, 5000 samples |
| PhysioWave | frozen wavelet transformer, trained decoder | 0.77 M | 12 leads, 2048 samples |
| QTcNet | direct regression, released weights | none | 12 leads, 100 Hz |

