# Dataset status — Brain Tumor MRI Dataset

- Source: Kaggle, Masoud Nickparvar, "Brain Tumor MRI Dataset"
  https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- Route used: direct Kaggle download endpoint (no account needed for this
  public dataset), `scripts/download_data.py` reproduces it.
- Downloaded: 2026-10-28. Dataset version 2 (Kaggle API reports
  currentVersionNumber 2, lastUpdated 2026-02-13). Zip: 164,670,110 bytes.

## Verified counts (counted from files on disk, 2026-10-28)

| Split | glioma | meningioma | notumor | pituitary | total |
|---|---|---|---|---|---|
| Training | 1,400 | 1,400 | 1,400 | 1,400 | 5,600 |
| Testing  | 400 | 400 | 400 | 400 | 1,600 |
| **Total** | 1,800 | 1,800 | 1,800 | 1,800 | **7,200** |

Note: the older published figures for this dataset (7,023 images;
5,712 Training / 1,311 Testing, unbalanced classes) describe version 1.
Version 2 on disk is balanced. All training/evaluation in this repo uses
the version-2 files counted above.

## Integrity checks (md5, 2026-10-28)

- 7,200 files, 7,013 unique md5 hashes -> 187 files are exact duplicates
  of another file inside the same split.
- Exact-hash overlap between Training and Testing: 0 (no identical file
  appears in both splits).
- Images vary in size and colour mode (e.g. 512x512 greyscale, 201x251 RGB);
  the loader converts everything to RGB and resizes to 224x224.
- The dataset ships no patient IDs, so splits are image-level only.
- Meningioma is the only class with augmented files, visible from the
  names: Training has 100 `Tr-aug-me_*` + 1,300 `Tr-me_*`; Testing has
  103 `Te-aug-me_*` + 297 `Te-me_*`. All other classes have a single
  prefix per split (`Tr-gl/Te-gl`, `Tr-no/Te-no`, `Tr-pi/Te-pi`).
