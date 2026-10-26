"""False-alarm check on REAL eye footage where no drop ever falls.

Slides a 12-frame window over the Pexels eye close-ups (blinks, looking around, camera
moves) and counts how often the model says "drop detected". Every window should be not detected.
"""
import glob
import json
import os

import cv2
import numpy as np
import onnxruntime as ort

from synth import H, T, W

MIN_DETECT = 0.7  # same cutoff as serve.py

ROOT = os.path.dirname(os.path.abspath(__file__))


def frames_of(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        # center crop to 4:3 then shrink
        cw = min(w, int(h * W / H))
        ch = int(cw * H / W)
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        f = cv2.resize(f[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_AREA)
        out.append(f[:, :, ::-1])
    return np.stack(out)


def main():
    sess = ort.InferenceSession(f"{ROOT}/runs/dropnet.onnx", providers=["CPUExecutionProvider"])
    res = {}
    allp = []
    for p in sorted(glob.glob(f"{ROOT}/data/raw/pexels/eye/*.mp4")):
        fr = frames_of(p)
        wins = [fr[i:i + T] for i in range(0, len(fr) - T, 4)]
        x = np.stack(wins).transpose(0, 1, 4, 2, 3).astype(np.float32)
        pr = np.concatenate([sess.run(None, {"clip": x[i:i + 32]})[0] for i in range(0, len(x), 32)])
        det = pr[:, 0] >= MIN_DETECT
        allp.append(det)
        res[os.path.basename(p)] = {"windows": len(det), "frac_detected": float(det.mean())}
    allp = np.concatenate(allp)
    summary = {"windows": int(len(allp)), "frac_detected": float(allp.mean()),
               "note": "real footage, no drop in any window, so frac_detected = false credit rate"}
    print(json.dumps(res, indent=1))
    print("SUMMARY", json.dumps(summary, indent=1))
    json.dump({"per_video": res, "summary": summary}, open(f"{ROOT}/runs/real_nodrop_metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
