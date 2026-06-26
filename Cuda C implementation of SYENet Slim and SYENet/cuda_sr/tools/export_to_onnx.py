#!/usr/bin/env python3
"""
Rebuild the recovered architecture as a torch.nn.Module, load the weights
straight from the unzipped checkpoint's raw storages (no torch.load zip
gymnastics), and export to ONNX.

Requires: torch, numpy. Run from anywhere:
    python export_to_onnx.py [--height 360] [--width 640] [--out model.onnx]

Module/parameter names are chosen so state_dict keys match the checkpoint
exactly (head.bias, head.block1.0.weight, ..., tail.1.bias) — load is strict.
"""
import argparse, os
import numpy as np
import torch
import torch.nn as nn

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

# (state_dict key, storage file, shape)
TENSORS = [
    ("head.bias",            "0",  (1, 12, 1, 1)),
    ("head.block1.0.weight", "1",  (12, 4, 5, 5)),
    ("head.block1.0.bias",   "2",  (12,)),
    ("head.block1.1.weight", "3",  (12,)),
    ("head.block1.2.weight", "4",  (12, 12, 3, 3)),
    ("head.block1.2.bias",   "5",  (12,)),
    ("head.block2.weight",   "6",  (12, 4, 5, 5)),
    ("head.block2.bias",     "7",  (12,)),
    ("body.bias",            "8",  (1, 12, 1, 1)),
    ("body.block1.weight",   "9",  (12, 12, 3, 3)),
    ("body.block1.bias",     "10", (12,)),
    ("body.block2.weight",   "11", (12, 12, 1, 1)),
    ("body.block2.bias",     "12", (12,)),
    ("att.1.weight",         "13", (12, 12, 1, 1)),
    ("att.1.bias",           "14", (12,)),
    ("att.2.weight",         "15", (12,)),
    ("att.3.weight",         "16", (12, 12, 1, 1)),
    ("att.3.bias",           "17", (12,)),
    ("tail.1.weight",        "18", (3, 3, 3, 3)),
    ("tail.1.bias",          "19", (3,)),
]


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, 12, 1, 1))
        self.block1 = nn.Sequential(
            nn.Conv2d(4, 12, 5, padding=2), nn.PReLU(12),
            nn.Conv2d(12, 12, 3, padding=1))
        self.block2 = nn.Conv2d(4, 12, 5, padding=2)

    def forward(self, x):
        return self.block1(x) + self.block2(x) + self.bias


class Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, 12, 1, 1))
        self.block1 = nn.Conv2d(12, 12, 3, padding=1)
        self.block2 = nn.Conv2d(12, 12, 1)

    def forward(self, x):
        # NOTE: ReLU and the residual connection are inferred (see README).
        return self.block2(torch.relu(self.block1(x))) + self.bias + x


class SRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = Head()
        self.body = Body()
        # att.0 = AdaptiveAvgPool2d (paramless), sigmoid applied in forward
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(12, 12, 1), nn.PReLU(12),
            nn.Conv2d(12, 12, 1))
        self.tail = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(3, 3, 3, padding=1))

    def forward(self, x):
        h = self.head(x)
        b = self.body(h)
        f = b * torch.sigmoid(self.att(b))
        return self.tail(f)


def load_state_dict():
    sd = {}
    for key, sfile, shape in TENSORS:
        raw = np.fromfile(os.path.join(CKPT, "data", sfile), dtype="<f4")
        assert raw.size == int(np.prod(shape)), key
        sd[key] = torch.from_numpy(raw.reshape(shape).copy())
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--out", default="model.onnx")
    args = ap.parse_args()

    net = SRNet().eval()
    missing, unexpected = net.load_state_dict(load_state_dict(), strict=True), None
    print("state_dict loaded strictly: OK")

    dummy = torch.rand(1, 4, args.height, args.width)
    torch.onnx.export(
        net, dummy, args.out,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h2", 3: "w2"}},
        opset_version=17)
    print(f"Exported {args.out}")

    # sanity: torch output vs reloaded onnx (if onnxruntime available)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        y_ort = sess.run(None, {"input": dummy.numpy()})[0]
        y_t = net(dummy).detach().numpy()
        print(f"torch vs onnxruntime max abs err: {np.abs(y_ort - y_t).max():.3e}")
    except ImportError:
        print("onnxruntime not installed — skipped parity check")


if __name__ == "__main__":
    main()
