"""Runs on the minicomputer (Raspberry Pi 5 + Camera Module 3 / any picamera2 cam).

Keeps a rolling buffer of small frames. When something moves fast in the frame (a drop
coming off the nozzle, a blink, a hand), it grabs 4 frames before + 8 after and sends
that clip to the server. --local runs the model on the Pi instead.

  python3 capture.py --server http://<server-ip>:8800
  python3 capture.py --local ../runs/dropnet.onnx
  python3 capture.py --webcam 0 --server ...   # test on a laptop without a pi
"""
import argparse
import io
import json
import time
import urllib.request
from collections import deque

import cv2
import numpy as np

W, H, T = 128, 96, 12
PRE = 4
FPS = 60  # drop falls ~2 cm in ~65 ms, so we need at least 60fps to see it
COOLDOWN_S = 1.5


def open_camera(webcam):
    if webcam is not None:
        cap = cv2.VideoCapture(webcam)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        return lambda: cap.read()[1][:, :, ::-1]
    from picamera2 import Picamera2  # only on the pi

    cam = Picamera2()
    cfg = cam.create_video_configuration(main={"size": (640, 480), "format": "RGB888"},
                                         controls={"FrameRate": FPS})
    cam.configure(cfg)
    cam.start()
    # picamera2 RGB888 comes out as BGR in memory
    return lambda: cam.capture_array()[:, :, ::-1]


def small(f):
    return cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)


def classify_remote(server, clip):
    buf = io.BytesIO()
    np.save(buf, clip)
    req = urllib.request.Request(f"{server}/classify", data=buf.getvalue(), method="POST",
                                 headers={"Content-Type": "application/octet-stream"})
    return json.loads(urllib.request.urlopen(req, timeout=5).read())


def make_local(path):
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    labels = ["went_in", "missed", "no_drop"]

    def run(clip):
        z = sess.run(None, {"clip": clip.transpose(0, 3, 1, 2)[None].astype(np.float32)})[0][0]
        p = np.exp(z - z.max())
        p /= p.sum()
        return {"verdict": labels[int(p.argmax())], "probs": dict(zip(labels, p.round(3).tolist()))}

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server")
    ap.add_argument("--local")
    ap.add_argument("--webcam", type=int)
    ap.add_argument("--motion", type=float, default=12.0, help="trigger threshold, mean abs diff in the hot spot")
    a = ap.parse_args()
    if not (a.server or a.local):
        ap.error("need --server or --local")

    grab = open_camera(a.webcam)
    run = make_local(a.local) if a.local else (lambda c: classify_remote(a.server, c))
    buf = deque(maxlen=T + PRE)
    prev = None
    last_fire = 0.0
    pending = None  # frames left to collect after a trigger

    while True:
        f = small(grab())
        buf.append(f)
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.int16)
        if prev is not None and pending is None and time.time() - last_fire > COOLDOWN_S:
            d = np.abs(g - prev)
            # max over a coarse grid so a small drop isn't averaged away by the whole frame
            score = cv2.resize(d.astype(np.float32), (8, 6), interpolation=cv2.INTER_AREA).max()
            if score > a.motion:
                pending = T - PRE
        elif pending is not None:
            pending -= 1
            if pending <= 0:
                clip = np.stack(list(buf)[-T:])
                last_fire = time.time()
                pending = None
                try:
                    print(time.strftime("%H:%M:%S"), run(clip), flush=True)
                except Exception as e:
                    print("classify failed:", e, flush=True)
        prev = g


if __name__ == "__main__":
    main()
