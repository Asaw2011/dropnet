"""Drop classifier server. The Pi posts a clip, gets back drop_detected / drop_not_detected.

  python serve.py --port 8800 [--forward http://localhost:4321]

POST /classify   body = .npy of uint8 [12, 96, 128, 3] RGB   -> {"verdict", "probs", "clip_id"}
POST /label      {"clip_id", "label"}  saves the true label so real clips can retrain the model
GET  /health

Every clip gets saved to data/collected/ so we build the real dataset as we go.
--forward tells the companion app when a drop is detected. not detected isn't sent, the
app already flags a dose that never gets confirmed.
"""
import argparse
import io
import json
import os
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import onnxruntime as ort

from synth import H, T, W

ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE = f"{ROOT}/data/collected"
OUTPUTS = ["drop_detected", "drop_not_detected"]
# only call it detected when the model is pretty sure, a wrong credit is the bad mistake
MIN_DETECT = 0.7


class Classifier:
    def __init__(self, path):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def __call__(self, clip):
        x = clip.transpose(0, 3, 1, 2)[None].astype(np.float32)
        p = self.sess.run(None, {"clip": x})[0][0]
        return {l: float(v) for l, v in zip(OUTPUTS, p)}


def make_handler(clf, forward):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {"ok": True, "input": [T, H, W, 3]})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n)
            if self.path == "/classify":
                try:
                    clip = np.load(io.BytesIO(raw), allow_pickle=False)
                except Exception as e:
                    return self._json(400, {"error": f"bad npy: {e}"})
                if clip.shape != (T, H, W, 3):
                    return self._json(400, {"error": f"expected {(T, H, W, 3)}, got {list(clip.shape)}"})
                probs = clf(clip)
                verdict = "drop_detected" if probs["drop_detected"] >= MIN_DETECT else "drop_not_detected"
                clip_id = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
                os.makedirs(SAVE, exist_ok=True)
                np.savez_compressed(f"{SAVE}/{clip_id}.npz", clip=clip.astype(np.uint8),
                                    probs=json.dumps(probs), verdict=verdict)
                out = {"verdict": verdict, "probs": probs, "clip_id": clip_id}
                if forward and verdict == "drop_detected":
                    try:
                        req = urllib.request.Request(f"{forward}/api/drop", method="POST",
                                                     data=json.dumps({"scenario": "good"}).encode(),
                                                     headers={"Content-Type": "application/json"})
                        urllib.request.urlopen(req, timeout=3).read()
                        out["forwarded"] = True
                    except Exception as e:
                        out["forwarded"] = f"failed: {e}"
                return self._json(200, out)
            if self.path == "/label":
                b = json.loads(raw or b"{}")
                if b.get("label") not in OUTPUTS or not os.path.exists(f"{SAVE}/{b.get('clip_id')}.npz"):
                    return self._json(400, {"error": f"need clip_id that exists and label in {OUTPUTS}"})
                with open(f"{SAVE}/labels.jsonl", "a") as f:
                    f.write(json.dumps({"clip_id": b["clip_id"], "label": b["label"], "at": time.time()}) + "\n")
                return self._json(200, {"ok": True})
            self._json(404, {"error": "not found"})

        def log_message(self, fmt, *a):
            print(time.strftime("%H:%M:%S"), fmt % a)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--model", default=f"{ROOT}/runs/dropnet.onnx")
    ap.add_argument("--forward", default=None, help="companion app base url, e.g. http://localhost:4321")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(Classifier(a.model), a.forward))
    print(f"listening on {a.host}:{a.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
