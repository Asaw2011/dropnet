"""runs/dropnet.pt -> runs/dropnet.onnx (outputs [p detected, p not detected]), checks onnx matches torch"""
import os

import numpy as np
import onnxruntime as ort
import torch

from model import DetectedHead, DropNet
from synth import H, T, W

ROOT = os.path.dirname(os.path.abspath(__file__))

m = DropNet(pretrained=False)
m.load_state_dict(torch.load(f"{ROOT}/runs/dropnet.pt", map_location="cpu"))
m = DetectedHead(m).eval()
x = torch.rand(1, T, 3, H, W) * 255
out = f"{ROOT}/runs/dropnet.onnx"
torch.onnx.export(m, x, out, input_names=["clip"], output_names=["probs"], opset_version=17,
                  dynamic_axes={"clip": {0: "batch"}, "probs": {0: "batch"}}, dynamo=False)

sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
a = sess.run(None, {"clip": x.numpy()})[0]
b = m(x).detach().numpy()
print("max diff torch vs onnx:", float(np.abs(a - b).max()), "size MB:", round(os.path.getsize(out) / 1e6, 2))
