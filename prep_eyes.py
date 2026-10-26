"""Cut single-eye crops + masks out of the periorbital dataset.

Output: data/eyes/{name}_{L|R}.jpg and .png (mask), 4:3 (160x120), eye centered.
Mask values after remap: 0 skin/other, 1 brow, 2 sclera, 3 iris, 4 caruncle, 5 lid
"""
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "data/raw/periorbital/periorbital_dataset")
OUT = os.path.join(ROOT, "data/eyes")
W, H = 160, 120
CONTEXT = 2.2  # crop width = eye width * this
MIN_CONTEXT = 1.6  # tighter than this and the eye fills the frame, skip it


def crops_for(img, mask):
    opening = np.isin(mask, (2, 3, 4)).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(opening)
    # biggest two blobs should be the two eyes
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:2] + 1
    out = []
    for k in order:
        x, y, w, h, area = stats[k]
        if area < 150 or w < 25:
            continue
        cx, cy = x + w / 2, y + h / 2
        cw = w * CONTEXT
        # no mirror padding, it makes fake eyes. shrink the crop to fit instead
        ih, iw = img.shape[:2]
        cw = min(cw, ih * W / H, iw)
        if cw < w * MIN_CONTEXT:
            continue
        ch = cw * H / W
        x0 = int(np.clip(cx - cw / 2, 0, iw - cw))
        y0 = int(np.clip(cy - ch / 2, 0, ih - ch))
        ci = img[y0:y0 + int(ch), x0:x0 + int(cw)]
        cm = mask[y0:y0 + int(ch), x0:x0 + int(cw)]
        ci = cv2.resize(ci, (W, H), interpolation=cv2.INTER_AREA)
        cm = cv2.resize(cm, (W, H), interpolation=cv2.INTER_NEAREST)
        side_name = "L" if cx < img.shape[1] / 2 else "R"
        out.append((side_name, ci, cm))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    pairs = []
    for sub in ("cfd", "celeb"):
        for ip in sorted(glob.glob(f"{SRC}/{sub}_final_data/{sub}_output_images/*.jpg")):
            mp = ip.replace("_output_images", "_output_masks").replace(".jpg", ".png")
            if os.path.exists(mp):
                pairs.append((sub, ip, mp))
    kept = 0
    for sub, ip, mp in pairs:
        img = cv2.imread(ip)
        mask = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if img is None or mask is None or img.shape[:2] != mask.shape[:2]:
            continue
        base = os.path.splitext(os.path.basename(ip))[0]
        for side, ci, cm in crops_for(img, mask):
            cv2.imwrite(f"{OUT}/{base}_{side}.jpg", ci, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(f"{OUT}/{base}_{side}.png", cm)
            kept += 1
    print(f"{len(pairs)} photos -> {kept} eye crops in {OUT}")


if __name__ == "__main__":
    sys.exit(main())
