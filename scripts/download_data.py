"""Download and verify the Brain Tumor MRI dataset from Kaggle.

The Kaggle download endpoint used here serves the current dataset
version without an account for this public dataset:

    python scripts/download_data.py

Files land in data/raw/ (gitignored). The script counts every class
folder after extraction and prints the verified counts — the numbers
in README/REPORT come from this count, not from the Kaggle page.
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://www.kaggle.com/api/v1/datasets/download/"
       "masoudnickparvar/brain-tumor-mri-dataset")
REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    zip_path = REPO / "data" / "brain-tumor-mri-dataset.zip"
    if not (RAW / "Training").exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, zip_path)
        print(f"extracting -> {RAW}")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(RAW)
        zip_path.unlink()
    total = 0
    for split in ("Training", "Testing"):
        for cls in CLASSES:
            folder = RAW / split / cls
            if not folder.is_dir():
                print(f"MISSING: {folder}", file=sys.stderr)
                return 1
            n = sum(1 for p in folder.iterdir() if p.is_file())
            total += n
            print(f"{split}/{cls}: {n}")
    print(f"TOTAL: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
