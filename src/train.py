"""Train NeuroNet on the Brain Tumor MRI Training split.

Model selection uses a stratified validation slice carved out of the
Training folder (see src/data_brain.py); the Testing folder is never
read here. A checkpoint is saved after EVERY epoch (last.pt), the best
validation model is kept separately (best.pt), and --resume continues
from last.pt — a restart can never lose more than the current epoch.

Usage:
    python -m src.train --epochs 15 --img-size 224
    python -m src.train --epochs 15 --resume
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.common.seed import set_seed
from src.data_brain import (BrainDataset, eval_transforms, resolve_root,
                            split_train_val, train_transforms)
from src.models import NeuroNet

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out", default="neurovision_resnet18")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += crit(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / total, correct / total


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu")
    root = resolve_root()
    train_items, val_items = split_train_val(root, args.val_frac, args.seed)
    print(f"train {len(train_items)}  val {len(val_items)}", flush=True)

    train_ds = BrainDataset(train_items, train_transforms(args.img_size))
    val_ds = BrainDataset(val_items, eval_transforms(args.img_size))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)

    model = NeuroNet(num_classes=4, pretrained=True)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params):,}", flush=True)
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    CKPT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    last_path = CKPT_DIR / f"{args.out}_last.pt"
    best_path = CKPT_DIR / f"{args.out}_best.pt"
    log_path = RESULTS_DIR / "train_log.csv"

    start_epoch, best_acc = 0, -1.0
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1
        best_acc = ck.get("best_acc", -1.0)
        print(f"resumed at epoch {start_epoch}, best val acc {best_acc:.4f}", flush=True)

    new_log = not log_path.exists() or start_epoch == 0
    log_f = open(log_path, "a", newline="")
    writer = csv.writer(log_f)
    if new_log:
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"])

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        # Frozen blocks must stay in eval mode so their BatchNorm
        # statistics do not drift while their weights are frozen.
        model.stem.eval(); model.layer1.eval(); model.layer2.eval()
        loss_sum, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        sched.step()
        train_loss, train_acc = loss_sum / total, correct / total
        val_loss, val_acc = evaluate(model, val_loader, device)
        secs = time.time() - t0
        print(f"epoch {epoch}: train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | {secs:.0f}s", flush=True)
        writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}",
                         f"{val_loss:.4f}", f"{val_acc:.4f}", f"{secs:.0f}"])
        log_f.flush()

        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "epoch": epoch,
                 "best_acc": max(best_acc, val_acc), "args": vars(args)}
        torch.save(state, last_path)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_acc": val_acc, "args": vars(args)}, best_path)
            print(f"  new best -> {best_path.name} (val acc {val_acc:.4f})", flush=True)

    log_f.close()
    meta = {"best_val_acc": best_acc, "epochs_run": args.epochs,
            "train_images": len(train_items), "val_images": len(val_items)}
    (RESULTS_DIR / "train_summary.json").write_text(json.dumps(meta, indent=2))
    print("done:", meta, flush=True)


if __name__ == "__main__":
    main()
