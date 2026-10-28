# NeuroVision — Brain Tumor MRI Classification

**A deep-learning classifier that reads a brain MRI scan and tells apart
four cases: glioma, meningioma, pituitary tumor, and no tumor — with a
Grad-CAM heatmap showing which part of the scan drove the decision.**

Brain tumors are hard to catch early and expensive to miss. A radiologist
reads hundreds of scans; a model like this one is a second pair of eyes —
it pre-screens a scan in a second and points at the region it found
suspicious. This project is that second pair of eyes, built end to end:
data, training, honest evaluation, explanations, and a small web app.

> **Research/educational tool — not a medical diagnosis.** Only a
> qualified clinician with the full patient history and the original
> scans can make a diagnosis.

This is Project 4 of my portfolio, following VisionAI (industrial defect
detection) and ForgeAD (self-supervised anomaly detection). It takes the
same recipe — transfer learning, Grad-CAM, a Flask dashboard, measured
metrics reported as-is — into medical imaging.

## Results (measured, held-out Testing split, 1,600 images)

| Metric | Value |
|---|---|
| Accuracy | 0.9563 |
| Macro-F1 | 0.9557 |
| AUROC (macro, one-vs-rest) | 0.9882 |

Per-class recall (sensitivity):

| Class | Recall | Precision | F1 | AUROC |
|---|---|---|---|---|
| Glioma | 0.8675 | 0.9830 | 0.9216 | 0.9637 |
| Meningioma | 0.9800 | 0.9356 | 0.9573 | 0.9933 |
| No Tumor | 1.0000 | 0.9259 | 0.9615 | 0.9970 |
| Pituitary Tumor | 0.9775 | 0.9874 | 0.9824 | 0.9988 |

The honest weak spot is glioma: 53 of its 400 test scans are missed, 29
of them predicted as "no tumor" — the error direction that matters most
in medicine. `REPORT.md` discusses this instead of hiding it.

All numbers come from `python -m src.evaluate` on the dataset's own
Testing split, which the model never trained on. Full details, the
confusion matrix, and an honest discussion of the limitations are in
`REPORT.md` and `results/`.

## How it works

1. **Data** (`src/data_brain.py`) — the Brain Tumor MRI Dataset by Masoud
   Nickparvar (Kaggle). I use the dataset's own Training/Testing split;
   a stratified 15% validation slice is carved out of Training for model
   selection, and Testing is only read by the evaluation script.
   Verified on disk (version 2): 5,600 Training + 1,600 Testing images,
   perfectly balanced at 1,400/400 per class — see `data/DATASET_STATUS.md`.
2. **Learn** (`src/train.py`) — a ResNet-18 pretrained on ImageNet,
   fine-tuned at 224×224 with flip/rotation/zoom augmentation. The early
   layers (stem through layer2) stay frozen to keep CPU training
   feasible — a documented shortcut, see `REPORT.md`. A checkpoint is
   saved after every epoch and training supports `--resume`.
3. **Explain** (`src/gradcam.py`) — Grad-CAM on the last convolutional
   layer highlights the scan regions behind each prediction.
4. **Evaluate** (`src/evaluate.py`) — accuracy, macro-F1, one-vs-rest
   AUROC, per-class recall/precision/F1, confusion matrix, and one
   Grad-CAM example per class, all on the held-out Testing split.
5. **Serve** (`app/app.py`) — a Flask web app: upload an MRI image and
   get the predicted class, per-class confidence bars, and the Grad-CAM
   overlay, plus a metrics page with the test results above.

## Quickstart

```bash
# 1. environment (CPU-only)
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. data (Brain Tumor MRI Dataset, Kaggle — downloaded separately)
python scripts/download_data.py

# 3. train (12 epochs, ~1.5 h on a 2-core CPU; resumes with --resume)
python -m src.train --epochs 12

# 4. evaluate on the held-out Testing split
python -m src.evaluate

# 5. web app
python app/app.py        # http://127.0.0.1:5000

# 6. single image from the command line
python -m src.infer data/raw/Testing/glioma/Te-gl_1.jpg
```

## Repository layout

```
neurovision-brain-tumor/
├── src/
│   ├── data_brain.py    # dataset loading, splits, transforms
│   ├── models.py        # ResNet-18 + 4-class head
│   ├── train.py         # training (per-epoch checkpoints, --resume)
│   ├── evaluate.py      # held-out test metrics + confusion matrix + Grad-CAM examples
│   ├── gradcam.py       # Grad-CAM (Selvaraju et al., ICCV 2017)
│   ├── infer.py         # single-image prediction (app + CLI)
│   └── common/seed.py
├── app/                 # Flask web app (templates + static)
├── scripts/download_data.py
├── tests/test_smoke.py
├── data/DATASET_STATUS.md   # provenance + verified counts
├── results/             # metrics JSON, plots, logs (generated)
├── REPORT.md            # research-style write-up
└── requirements.txt
```

## Dataset citation

Masoud Nickparvar, "Brain Tumor MRI Dataset", Kaggle, 2021.
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
(The dataset combines figshare, SARTAJ and Br35H sources; it is used here
for research/education and is not redistributed in this repository.)

## References

* K. He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
* R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
  via Gradient-based Localization", ICCV 2017.
* J. Cheng et al., "Enhanced Performance of Brain Tumor Classification via
  Tumor Region Augmentation and Partition", PLOS ONE 2015 (figshare source).

## Author

Muhammad Arshaad — BS Information Technology, Government College
University Faisalabad. Portfolio: https://github.com/marshaad708-coder

## License

Code: MIT. The dataset is downloaded separately from Kaggle under its own
terms and is not shipped here.
