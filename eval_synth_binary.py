"""Detected / not detected scores on the held-out synthetic test set, using the shipped ONNX."""
import json
import os

import numpy as np
import onnxruntime as ort

from train import SynthClips, split_eyes

ROOT = os.path.dirname(os.path.abspath(__file__))
MIN_DETECT = 0.7  # same cutoff as serve.py

if __name__ == "__main__":
    sess = ort.InferenceSession(f"{ROOT}/runs/dropnet.onnx", providers=["CPUExecutionProvider"])
    ds = SynthClips(split_eyes()["test"], 3000, seed=2)  # same clips train.py tested on
    p, y = [], []
    for i in range(len(ds)):
        x, lab = ds[i]
        p.append(sess.run(None, {"clip": x[None].numpy()})[0][0, 0])
        y.append(lab)
    p, y = np.array(p), np.array(y)
    det, went_in = p >= MIN_DETECT, y == 0
    m = {
        "accuracy": float((det == went_in).mean()),
        "detected_when_it_went_in": float(det[went_in].mean()),
        "detected_when_it_missed": float(det[y == 1].mean()),
        "detected_when_nothing_fell": float(det[y == 2].mean()),
        "cutoff": MIN_DETECT,
        "note": "synthetic clips on held-out faces, NOT real drops",
    }
    print(json.dumps(m, indent=1))
    json.dump(m, open(f"{ROOT}/runs/test_metrics_binary.json", "w"), indent=1)
