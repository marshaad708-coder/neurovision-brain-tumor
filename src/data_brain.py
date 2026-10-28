"""Brain Tumor MRI dataset loading.

Dataset: Masoud Nickparvar, "Brain Tumor MRI Dataset", Kaggle.
The copy on disk (dataset version 2, downloaded 2026-10-28) has folders:

    data/raw/Training/{glioma,meningioma,notumor,pituitary}/
    data/raw/Testing/{glioma,meningioma,notumor,pituitary}/

Verified on disk: 1,400 Training + 400 Testing images per class
(5,600 / 1,600, 7,200 total) — see data/DATASET_STATUS.md. This differs
from the older published counts (7,023); the counts here are the ones
counted from the actual files, and those are what the code uses.

The provided Training/Testing split is kept as-is. A stratified
validation set (15%) is carved out of Training for model selection;
Testing is only touched by src/evaluate.py.
"""
import os
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Folder names on disk and the names shown to people.
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def resolve_root() -> Path:
    env = os.environ.get("NEUROVISION_DATA")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "data" / "raw"


def list_images(root: Path, split: str) -> list[tuple[Path, int]]:
    """All (path, label) pairs of one split, sorted for reproducibility."""
    out = []
    for label, cls in enumerate(CLASS_NAMES):
        folder = root / split / cls
        if not folder.is_dir():
            raise FileNotFoundError(f"missing dataset folder: {folder}")
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                out.append((p, label))
    return out


def split_train_val(root: Path, val_frac: float = 0.15, seed: int = 7):
    """Stratified train/val split carved out of the Training folder."""
    rng = random.Random(seed)
    by_class: dict[int, list] = {}
    for p, y in list_images(root, "Training"):
        by_class.setdefault(y, []).append(p)
    train, val = [], []
    for y, paths in sorted(by_class.items()):
        paths = list(paths)
        rng.shuffle(paths)
        n_val = int(round(len(paths) * val_frac))
        val += [(p, y) for p in paths[:n_val]]
        train += [(p, y) for p in paths[n_val:]]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def train_transforms(img_size: int = 224):
    """Augmentation for training: flips, small rotations, small zoom.

    MRI scans have no canonical left/right orientation in this dataset,
    so horizontal flipping is safe. Rotations stay small (a head in a
    scanner is roughly upright) and zoom is a random resized crop.
    """
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transforms(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class BrainDataset(Dataset):
    def __init__(self, items, tf):
        self.items = items
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        with Image.open(path) as im:
            img = im.convert("RGB")
        return self.tf(img), y


def load_rgb(path, img_size: int) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB").resize((img_size, img_size))
