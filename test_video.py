"""Run the model on your own videos of real drops and see what it says.

  python test_video.py videos/*.mov
  python test_video.py clip.mov --crop 0.2,0.3,0.5,0.4    # x,y,w,h as fractions of the frame

Name files by what really happened and it'll score itself:
  in_*.mov    drop went in
  miss_*.mov  drop missed (lid, corner, cheek, closed eye)
  none_*.mov  no drop came out
Anything else just gets a verdict.

For each video it writes to runs/video_tests/:
  <name>_marked.mp4  the video with the model's score drawn on every frame (plays slowed down)
  <name>_peak.png    the 12 frames it was most sure about
and prints a summary table at the end.

Film close, eye filling most of the frame, slow-mo if you can (frames get thinned to ~60 fps
to match what the model trained on). Saline / artificial tears only.
"""
import argparse
import json
import os
import shutil
import subprocess

import cv2
import numpy as np
import onnxruntime as ort

from synth import H, T, W

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = f"{ROOT}/runs/video_tests"
MIN_DETECT = 0.7  # same cutoff as serve.py
MODEL_FPS = 60
TRUTH = {"in": "went in", "miss": "missed", "none": "nothing fell"}


def read_video(path, crop):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(fps / MODEL_FPS))  # 240fps slo-mo -> keep every 4th
    small, full, i = [], [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        i += 1
        if (i - 1) % step:
            continue
        h, w = f.shape[:2]
        if crop:
            x, y, cw, ch = crop
            f = f[int(y * h):int((y + ch) * h), int(x * w):int((x + cw) * w)]
            h, w = f.shape[:2]
        # center crop to 4:3 like the training clips
        cw = min(w, int(h * W / H))
        ch = int(cw * H / W)
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        f = f[y0:y0 + ch, x0:x0 + cw]
        full.append(cv2.resize(f, (480, 360), interpolation=cv2.INTER_AREA))
        small.append(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)[:, :, ::-1])
    return np.stack(small), full, fps / step


def score(sess, frames):
    """p(detected) for every 12-frame window, stride 1"""
    starts = list(range(0, max(1, len(frames) - T + 1)))
    if len(frames) < T:
        return np.zeros(0), starts
    x = np.stack([frames[s:s + T] for s in starts]).transpose(0, 1, 4, 2, 3).astype(np.float32)
    p = np.concatenate([sess.run(None, {"clip": x[i:i + 32]})[0][:, 0] for i in range(0, len(x), 32)])
    return p, starts


def write_marked(path, full, p, fps):
    # score shows up on a window's last frame, that's when the device would know
    per_frame = np.zeros(len(full))
    for s, v in enumerate(p):
        per_frame[s + T - 1] = v
    tmp = path + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), min(fps, 30), (480, 420))
    for i, f in enumerate(full):
        canvas = np.full((420, 480, 3), 255, np.uint8)
        canvas[:360] = f
        # score timeline along the bottom
        xs = np.linspace(0, 479, len(per_frame)).astype(int)
        ys = (415 - per_frame * 50).astype(int)
        cy = int(415 - MIN_DETECT * 50)
        cv2.line(canvas, (0, cy), (479, cy), (180, 180, 180), 1)
        cv2.polylines(canvas, [np.stack([xs, ys], 1).reshape(-1, 1, 2)], False, (90, 90, 90), 1)
        cv2.line(canvas, (xs[i], 365), (xs[i], 419), (0, 0, 0), 1)
        v = per_frame[i]
        det = v >= MIN_DETECT
        col = (40, 150, 40) if det else (60, 60, 60)
        cv2.rectangle(canvas, (0, 0), (479, 30), (255, 255, 255), -1)
        cv2.putText(canvas, f"{'DROP DETECTED' if det else 'not detected'}  p={v:.2f}", (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        vw.write(canvas)
    vw.release()
    # opencv's mp4v won't open in quicktime, re-encode to h264 if ffmpeg is around
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        path], check=False)
    if os.path.exists(path):
        os.remove(tmp)
    else:
        os.rename(tmp, path)


def write_peak(path, frames, p, title):
    k = int(p.argmax())
    strip = np.hstack([cv2.resize(f[:, :, ::-1], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                       for f in frames[k:k + T]])
    bar = np.full((34, strip.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.imwrite(path, np.vstack([bar, strip]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--crop", help="x,y,w,h as fractions, to zoom in on the eye")
    a = ap.parse_args()
    crop = [float(v) for v in a.crop.split(",")] if a.crop else None
    sess = ort.InferenceSession(f"{ROOT}/runs/dropnet.onnx", providers=["CPUExecutionProvider"])
    os.makedirs(OUT, exist_ok=True)

    rows = []
    for path in a.videos:
        name = os.path.splitext(os.path.basename(path))[0]
        frames, full, fps = read_video(path, crop)
        p, starts = score(sess, frames)
        if len(p) == 0:
            print(f"{name}: too short, need at least {T} frames at ~{MODEL_FPS} fps")
            continue
        peak = float(p.max())
        detected = peak >= MIN_DETECT
        when = starts[int(p.argmax())] / fps
        truth = TRUTH.get(name.split("_")[0].lower())
        right = None if truth is None else detected == (truth == "went in")
        write_marked(f"{OUT}/{name}_marked.mp4", full, p, fps)
        write_peak(f"{OUT}/{name}_peak.png", frames, p,
                   f"{name}: {'DROP DETECTED' if detected else 'not detected'}, peak p={peak:.2f} at {when:.2f}s")
        rows.append({"video": name, "truth": truth, "verdict": "detected" if detected else "not detected",
                     "peak_p": round(peak, 3), "at_s": round(when, 2), "correct": right})
        mark = "" if right is None else ("  RIGHT" if right else "  WRONG")
        print(f"{name:30s} {'DROP DETECTED' if detected else 'not detected':14s} peak {peak:.2f} at {when:.2f}s"
              f"{'' if truth is None else '  (really: ' + truth + ')'}{mark}")

    scored = [r for r in rows if r["correct"] is not None]
    if scored:
        went = [r for r in scored if r["truth"] == "went in"]
        notin = [r for r in scored if r["truth"] != "went in"]
        print(f"\n{sum(r['correct'] for r in scored)}/{len(scored)} right")
        if went:
            print(f"drop went in, said detected:   {sum(r['verdict'] == 'detected' for r in went)}/{len(went)}")
        if notin:
            print(f"didn't go in, said detected:   {sum(r['verdict'] == 'detected' for r in notin)}/{len(notin)}"
                  "  <- this one should be 0")
    json.dump(rows, open(f"{OUT}/results.json", "w"), indent=1)
    print(f"\nmarked videos + peak frames in {OUT}")


if __name__ == "__main__":
    main()
