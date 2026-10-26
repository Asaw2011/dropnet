# Dropwise camera model

A small neural network that watches about 0.2 s of video from the clip's camera and answers one question:
**drop detected** (a drop went into the eye) or **drop not detected** (it missed, hit a closed eye, or nothing fell).

It calls it detected only when it's at least 70% sure (`MIN_DETECT` in `serve.py`), because giving credit for a missed dose is the worse mistake.

Camera only. No beam-break or thermopile.

```
pi/capture.py  (Raspberry Pi + camera, 60 fps)  --12-frame clip-->  serve.py (server, ONNX)  --drop detected-->  companion app /api/drop
```

## Why the training data is simulated
There's no public video of eye drops filmed from a bottle camera (see `SOURCES.md` for what I checked).
So for now the model trains on **real eye photos with simulated drops**:

1. `prep_eyes.py` cuts 5,617 single-eye crops out of the periorbital dataset (2,842 faces, CC BY 4.0). Every crop has a hand-drawn mask of the sclera, iris, and caruncle, so we know exactly which pixels count as "in the eye".
2. `synth.py` builds each 12-frame clip. A drop comes in big from the nozzle side, shrinks as it falls, then:
   - **went_in**: it lands inside the mask, spreads into a glossy tear film, the lower lid line lights up, and often there's a blink. Sometimes a little overflows.
   - **missed**: it lands on skin (near misses within a few pixels of the lid edge, or far), beads up, and often runs down. In 15% of misses the eye closes just as the drop arrives, so it hits the lid. These force the model to look at whether the eye is open.
   - **no_drop**: nothing, a blink, or a drop that hangs on the nozzle and never falls.
   The base images also get random zoom, tilt, color shifts, grayscale (NoIR camera), blur, hand shake, flicker and sensor noise.
   Training keeps these three cases apart so the model learns what a near miss looks like. The shipped model (`DetectedHead` in `model.py`) merges missed and no_drop into "not detected".
3. `train.py` trains `DropNet` (MobileNetV3-small on each frame, then a temporal conv over the 12 frames, 1.3M params). Faces are split 80/10/10 **by photo**, so test faces are never seen in training.

## Commands
```
uv venv -p 3.12 .venv && uv pip install -p .venv torch torchvision opencv-python-headless numpy pillow onnx onnxruntime scikit-learn tqdm
.venv/bin/python prep_eyes.py
.venv/bin/python train.py --epochs 12          # ~55 min on an M-series Mac
.venv/bin/python export_onnx.py                # runs/dropnet.onnx, outputs [p detected, p not detected]
.venv/bin/python eval_synth_binary.py          # detected / not detected scores on the simulated test set
.venv/bin/python eval_real_nodrop.py           # false alarms on real eye video
.venv/bin/python serve.py --forward http://localhost:4321
```
On the Pi: `pip install opencv-python-headless numpy onnxruntime` (picamera2 ships with Raspberry Pi OS), copy `pi/capture.py` (and `runs/dropnet.onnx` for `--local`), then `python3 capture.py --server http://<server-ip>:8800`.
Testing without a Pi: `python pi/capture.py --webcam 0 --server http://localhost:8800`.

## What it can't claim yet
- **Every number below is on simulated drops.** A high score only means the model learned the simulator. It says nothing about real drops until we test on real clips from our camera.
- The camera angle, distance, lens, and lighting of the real clip aren't fixed yet. The simulator guesses: one whole eye in view, from a few cm.
- Real tear film, splash and runoff look different from the simulated ones. A real drop is also much clearer in the eye than in our renders. That gap is the main risk.

## Getting real data (the next step)
`serve.py` saves every clip it sees to `data/collected/`. Label them with `POST /label {"clip_id", "label": "drop_detected" | "drop_not_detected"}`, then run `finetune_real.py`.
Suggested first round, all with **preservative-free artificial tears or saline**, an adult volunteer, and an adult supervising:
- 3 or more people, both eyes, 2 lighting setups (lamp, daylight).
- About 60 clips where the drop goes in, and about 120 where it doesn't: half misses (aim at the lower lid, the outer corner, the cheek), half with no drop (squeeze an empty bottle, blink, move the bottle).
- Film every attempt from the side with a phone at the same time. The side video is the ground truth label.
- If this ever moves to patients (Ritu Dadi's glaucoma volunteers), that needs ethics approval and consent first.

## Results (2026-10-25, 12 epochs, ~55 min on M-series Mac)
Simulated test set: 3,000 clips on 566 held-out eye crops, 70% cutoff (`runs/test_metrics_binary.json`)
| | |
|---|---|
| Accuracy | 99.1% |
| Drop went in, said detected | 97.6% |
| Drop missed, said detected (wrong credit) | 0.3% |
| Nothing fell, said detected | 0% |

Real eye video with no drop (14 Pexels clips, 1,411 windows, `runs/real_nodrop_metrics.json`): said detected 0 times.
This is the only real-footage number so far, and it only tests false alarms from blinks and movement.

Model: 1.3M params, 5.1 MB ONNX, ~12 ms per clip on a Mac CPU (expect roughly 50-150 ms on a Pi 5, not measured).
