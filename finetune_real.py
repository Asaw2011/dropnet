"""Fine-tune on real labeled clips from the device (data/collected + labels.jsonl).

Mixes real clips with fresh synthetic ones so it doesn't forget, and holds out the
newest 20% of clips as the real-world test. todo: split by person once we log who it was

  python finetune_real.py --epochs 8
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from model import DropNet
from synth import LABELS
from train import SynthClips, evaluate, split_eyes

ROOT = os.path.dirname(os.path.abspath(__file__))
COL = f"{ROOT}/data/collected"


class RealClips(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cid, label = self.items[i]
        clip = np.load(f"{COL}/{cid}.npz")["clip"]
        return torch.from_numpy(clip).permute(0, 3, 1, 2).float(), LABELS.index(label)


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
    real = RealClips(tr_real)
    # oversample real so it's about half of every batch
    mix = ConcatDataset([real] * max(1, 1000 // len(real)) + [SynthClips(split_eyes()["train"], 1000)])
    tr = DataLoader(mix, batch_size=32, shuffle=True, num_workers=4)
    te = DataLoader(RealClips(te_real), batch_size=32)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5, 1.0], device=dev))
    print("before:", evaluate(m, te, dev)[0])
    for ep in range(a.epochs):
        m.train()
        for x, y in tr:
            loss = lossf(m(x.to(dev)), y.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"ep {ep + 1}", evaluate(m, te, dev)[0])
    torch.save(m.state_dict(), f"{ROOT}/runs/dropnet_real.pt")


if __name__ == "__main__":
    main()
