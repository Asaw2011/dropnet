# DropNet: the Dropwise camera model

A small neural network that watches about 0.2 s of video from the clip's camera and answers one question:
**drop detected** (a drop went into the eye) or **drop not detected** (it missed, hit a closed eye, or nothing fell).

It calls it detected only when it's at least 70% sure (`MIN_DETECT` in `serve.py`), because giving credit for a missed dose is the worse mistake.

Camera only. No beam-break or thermopile.

This repo is just the network. The companion app it reports to is [dropwise-app](https://github.com/Asaw2011/dropwise-app).

```
pi/capture.py  (Raspberry Pi + camera, 60 fps)  --12-frame clip-->  serve.py (server, ONNX)  --drop detected-->  companion app /api/drop
```

## Why the training data is simulated
There's no public video of eye drops filmed from a bottle camera (see `SOURCES.md` for what I checked).
So for now the model trains on **real eye photos with simulated drops**:

1. `prep_eyes.py` cuts 5,617 single-eye crops out of the periorbital dataset (2,842 faces, CC BY 4.0). Every crop has a hand-drawn mask of the sclera, iris, and caruncle, so we know exactly which pixels count as "in the eye".
2. `synth.py` builds each 12-frame clip. A drop comes in big from the nozzle side, shrinks as it falls, then:
   - **went_in**: most real drops don't hit the center, so only 30% land anywhere in the eye, 30% land on its edges and corners, and 40% land on the lid margin, inner/outer corner or lower lid and run in. Then it spreads into a glossy tear film, the lower lid line lights up, and often there's a blink. Sometimes a little overflows.
   - **missed**: it lands clearly on skin (brow, cheek, temple, upper lid well above the lashes), beads up, and often runs down. In 15% of misses the eye closes just as the drop arrives, so it hits the lid. A thin strip just outside the lid margin is too close to call, so no drop ever lands there.
   - **no_drop**: nothing, a blink, or a drop that hangs on the nozzle and never falls.
   Half the clips of every label also get a bottle tip at the top edge and some get a finger poking in, so a hand showing up means nothing on its own.
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

## Testing on your own videos
Easiest: `.venv/bin/python tester.py`, open http://localhost:8765, record or upload a video (phone slow-mo works best), drag a box around the eye, pick what really happened, hit Run test. It follows the eye through the video and crops to it, since the network expects a close-up. Videos land in `data/my_videos/` (gitignored).

From the command line, `test_video.py` does the same thing:

1. Film close, with the eye filling most of the frame, in slow-mo if you can. Use saline or preservative-free artificial tears only.
2. Name each file by what really happened: `in_1.mov` (went in), `miss_1.mov` (missed), `none_1.mov` (nothing came out).
3. Run `.venv/bin/python test_video.py ~/Desktop/drops/*.mov`. If the eye is small in the shot, add `--eye t,x,y,w,h`: a box around the eye (fractions of the frame) at t seconds, which gets tracked.

It prints each verdict, says whether it was right, and adds a summary. The line that matters most is "didn't go in, said detected", which should be 0.
In `runs/video_tests/` you get a marked copy of each video (score on every frame, slowed down) and a strip of the 12 frames it was most sure about.

## Getting real data (the next step)
`serve.py` saves every clip it sees to `data/collected/`. Label them with `POST /label {"clip_id", "label": "drop_detected" | "drop_not_detected"}`, then run `finetune_real.py`.
Suggested first round, all with **preservative-free artificial tears or saline**, an adult volunteer, and an adult supervising:
- 3 or more people, both eyes, 2 lighting setups (lamp, daylight).
- About 60 clips where the drop goes in, and about 120 where it doesn't: half misses (aim at the lower lid, the outer corner, the cheek), half with no drop (squeeze an empty bottle, blink, move the bottle).
- Film every attempt from the side with a phone at the same time. The side video is the ground truth label.
- If this ever moves to patients (Ritu Dadi's glaucoma volunteers), that needs ethics approval and consent first.

## Results (2026-11-01, v2, 12 epochs, ~55 min on M-series Mac)
v2 counts drops that land on the lid edge or a corner and run in as "went in", and adds a bottle tip / finger to every label.
Simulated test set: 3,000 clips on 566 held-out eye crops, 70% cutoff (`runs/test_metrics_binary.json`)
| | |
|---|---|
| Accuracy | 98.3% |
| Drop went in, said detected | 95.3% |
| Drop missed, said detected (wrong credit) | 0.1% |
| Nothing fell, said detected | 0% |

Drops that went in, by where they landed (same clips, `runs/test_by_landing_spot.json`):
| Landed | v1 | v2 |
|---|---|---|
| Anywhere in the eye | 92.2% | 98.4% |
| Edge of the eye | 94.2% | 99.7% |
| Lid margin / corner, then ran in | 44.8% | 89.5% |

v1 was trained to call that last case a miss, which is why real drops at the side of the eye weren't counted. v1's files are kept as `runs/*_v1.*`.

Real eye video with no drop (14 Pexels clips, 1,411 windows, `runs/real_nodrop_metrics.json`): said detected 0 times.

Real drops (2026-11-01, 2 went in + 1 missed, phone video of a whole face, eye box in `tester.py`): v2 gets the miss right but neither drop that went in (peaks 0.30 and 0.37 to 0.57 depending on the box). In both, the bottle tip is pressed right against the inner corner, so the drop never visibly falls, it goes straight from the tip into the corner behind it. The simulator doesn't make that case yet. Three videos is far too few to say much either way.

Model: 1.3M params, 5.1 MB ONNX, ~12 ms per clip on a Mac CPU (expect roughly 50-150 ms on a Pi 5, not measured).
