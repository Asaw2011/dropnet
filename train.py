"""Train DropNet on simulated clips. Splits are by photo, so test faces are never seen.

usage: python train.py [--epochs 12] [--per-epoch 8000]
"""
import argparse
import glob
import json
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import DropNet
from synth import LABELS, T, make_clip

ROOT = os.path.dirname(os.path.abspath(__file__))


def split_eyes(seed=0):
    files = sorted(glob.glob(f"{ROOT}/data/eyes/*.jpg"))
    # both eyes of one photo go in the same split
    photos = sorted({os.path.basename(f).rsplit("_", 1)[0] for f in files})
    random.Random(seed).shuffle(photos)
    n = len(photos)
    parts = {"train": set(photos[: int(0.8 * n)]), "val": set(photos[int(0.8 * n): int(0.9 * n)]),
             "test": set(photos[int(0.9 * n):])}
    return {k: [f for f in files if os.path.basename(f).rsplit("_", 1)[0] in v] for k, v in parts.items()}


class SynthClips(Dataset):
    def __init__(self, files, n, seed=None):
        self.files, self.n, self.seed = files, n, seed
        self.cache = {}

    def __len__(self):
        return self.n

    def _load(self, f):
        if f not in self.cache:
            self.cache[f] = (cv2.imread(f)[:, :, ::-1].copy(), cv2.imread(f[:-4] + ".png", 0))
        return self.cache[f]

    def __getitem__(self, i):
        # fixed seed per index for val/test so the sets never change
        rng = np.random.default_rng(None if self.seed is None else self.seed * 1_000_003 + i)
        f = self.files[rng.integers(len(self.files))]
        img, mask = self._load(f)
        clip, label = make_clip(img, mask, int(rng.integers(3)), rng)
        x = torch.from_numpy(clip).permute(0, 3, 1, 2).float()
        return x, label


def evaluate(model, loader, dev):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(dev)).softmax(1).cpu()
            ys.append(y)
            ps.append(p)
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    pred = p.argmax(1)
    cm = np.zeros((3, 3), int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    # binary view: did it go in or not
    yb, pb = y == 0, pred == 0
    return {
        "acc3": float((pred == y).mean()),
        "acc_binary": float((yb == pb).mean()),
        "sensitivity_went_in": float(pb[yb].mean()),
        "specificity_not_in": float((~pb[~yb]).mean()),
        "false_credit_on_miss": float(pb[y == 1].mean()),
        "confusion_rows_true_cols_pred": cm.tolist(),
    }, p, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--per-epoch", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    sp = split_eyes()
    print({k: len(v) for k, v in sp.items()}, "eye crops", "device", dev)

    tr = DataLoader(SynthClips(sp["train"], args.per_epoch), batch_size=args.bs, num_workers=args.workers,
                    shuffle=False, persistent_workers=True)
    va = DataLoader(SynthClips(sp["val"], 1500, seed=1), batch_size=64, num_workers=args.workers)
    te = DataLoader(SynthClips(sp["test"], 3000, seed=2), batch_size=64, num_workers=args.workers)

    model = DropNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = args.epochs * len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    # a missed drop counted as "went in" is the worst mistake, weight it up
    lossf = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5, 1.0], device=dev), label_smoothing=0.05)

    os.makedirs(f"{ROOT}/runs", exist_ok=True)
    best = -1
    for ep in range(args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            loss = lossf(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        m, _, _ = evaluate(model, va, dev)
        print(f"ep {ep + 1} loss {tot / n:.3f} val acc3 {m['acc3']:.3f} bin {m['acc_binary']:.3f} "
              f"false-credit {m['false_credit_on_miss']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if m["acc_binary"] > best:
            best = m["acc_binary"]
            torch.save(model.state_dict(), f"{ROOT}/runs/dropnet.pt")

    model.load_state_dict(torch.load(f"{ROOT}/runs/dropnet.pt", map_location=dev))
    m, _, _ = evaluate(model, te, dev)
    m["note"] = "synthetic clips on held-out faces, NOT real drops"
    print("TEST", json.dumps(m, indent=1))
    json.dump(m, open(f"{ROOT}/runs/test_metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
