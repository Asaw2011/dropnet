"""Fine-tune on real labeled clips from the device (data/collected + labels.jsonl).

Real labels are just drop_detected / drop_not_detected. The model still has 3 outputs
inside, so real clips train on went_in vs (missed + no_drop) merged, and synthetic clips
keep their 3 classes so it doesn't forget what a miss looks like.
Holds out the newest 20% of real clips as the test. todo: split by person once we log who it was

  python finetune_real.py --epochs 8
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import OUTPUTS, DropNet
from train import SynthClips, split_eyes

ROOT = os.path.dirname(os.path.abspath(__file__))
COL = f"{ROOT}/data/collected"
MIN_DETECT = 0.7  # same cutoff as serve.py


class RealClips(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cid, label = self.items[i]
        clip = np.load(f"{COL}/{cid}.npz")["clip"]
        # 0 detected, 1 not detected
        return torch.from_numpy(clip).permute(0, 3, 1, 2).float(), OUTPUTS.index(label)


def binary_loss(logits, y):
    p = logits.softmax(1)
    p_det = p[:, 0].clamp(1e-6, 1 - 1e-6)
    # a wrong credit is the bad mistake, weight not-detected clips up
    w = torch.where(y == 1, 1.5, 1.0)
    return (w * F.binary_cross_entropy(p_det, (y == 0).float(), reduction="none")).mean()


def eval_real(m, loader, dev):
    m.eval()
    ys, ds = [], []
    with torch.no_grad():
        for x, y in loader:
            ds.append(m(x.to(dev)).softmax(1)[:, 0].cpu() >= MIN_DETECT)
            ys.append(y == 0)
    y, d = torch.cat(ys), torch.cat(ds)
    return {"acc": float((y == d).float().mean()),
            "detected_when_it_went_in": float(d[y].float().mean()) if y.any() else None,
            "detected_when_it_didnt": float(d[~y].float().mean()) if (~y).any() else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    labels = {}
    for line in open(f"{COL}/labels.jsonl"):
        r = json.loads(line)
        labels[r["clip_id"]] = r["label"]  # last label wins if relabeled
    items = sorted((k, v) for k, v in labels.items() if os.path.exists(f"{COL}/{k}.npz"))
    if len(items) < 30:
        raise SystemExit(f"only {len(items)} labeled real clips, collect more first")
    cut = int(len(items) * 0.8)
    tr_real, te_real = items[:cut], items[cut:]
    print(len(tr_real), "train /", len(te_real), "test real clips")

    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    m = DropNet().to(dev)
    m.load_state_dict(torch.load(f"{ROOT}/runs/dropnet.pt", map_location=dev))
    real = DataLoader(RealClips(tr_real), batch_size=16, shuffle=True)
    synth = DataLoader(SynthClips(split_eyes()["train"], len(real) * 16), batch_size=16, num_workers=4)
    te = DataLoader(RealClips(te_real), batch_size=32)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
    cw = torch.tensor([1.0, 1.5, 1.0], device=dev)
    print("before:", eval_real(m, te, dev))
    for ep in range(a.epochs):
        m.train()
        for (xr, yr), (xs, ys) in zip(real, synth):
            loss = binary_loss(m(xr.to(dev)), yr.to(dev)) + F.cross_entropy(m(xs.to(dev)), ys.to(dev), weight=cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"ep {ep + 1}", eval_real(m, te, dev))
    torch.save(m.state_dict(), f"{ROOT}/runs/dropnet_real.pt")
    print("saved runs/dropnet_real.pt, copy it over dropnet.pt and rerun export_onnx.py to ship it")


if __name__ == "__main__":
    main()
