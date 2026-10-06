"""Compare a CLIP model on the CIX NPU (through Immich's own encoders, so preprocessing and tokenization are the
real ones) with its fp32 ONNX export on the CPU. Prints cosine similarity per modality and NPU latency.

Run from immich-v3/machine-learning with the venv:
  HF_HUB_OFFLINE=1 LD_LIBRARY_PATH=/usr/share/cix/lib PYTHONPATH=$PWD \
  MACHINE_LEARNING_CACHE_FOLDER=<cache with clip/<model>/> \
    .venv/bin/python <this> chinese-clip-vit-large-patch14 --onnx-image img.onnx --onnx-text txt.onnx --images list.txt
"""

import argparse
import time

import numpy as np
import onnxruntime as ort
from immich_ml.models.clip.textual import OpenClipTextualEncoder
from immich_ml.models.clip.visual import OpenClipVisualEncoder
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--onnx-image")
parser.add_argument("--onnx-text")
parser.add_argument("--images", required=True, help="file with one image path per line")
parser.add_argument(
    "--queries",
    default="海边日落,生日蛋糕,一只狗,猫在睡觉,雪山,夜景,婚礼,火锅,樱花,a dog on the beach,sunset,a red car",
)
args = parser.parse_args()


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


so = ort.SessionOptions()
so.log_severity_level = 3
vis, txt = OpenClipVisualEncoder(args.model), OpenClipTextualEncoder(args.model)
vis.load()
txt.load()
print("visual:", vis.session.get_inputs(), vis.session.get_outputs())
print("text:  ", txt.session.get_inputs(), txt.session.get_outputs())

if args.onnx_image:
    onnx = ort.InferenceSession(args.onnx_image, so, providers=["CPUExecutionProvider"])
    sims, times = [], []
    for path in open(args.images).read().split():
        x = vis.transform(Image.open(path).convert("RGB"))["image"]
        t = time.perf_counter()
        a = vis.session.run(None, {"image": x})[0][0]
        times.append(time.perf_counter() - t)
        sims.append(cos(a, onnx.run(None, {onnx.get_inputs()[0].name: x})[0][0]))
    print(
        f"image NPU vs ONNX cos: min={min(sims):.4f} mean={np.mean(sims):.4f} | "
        f"NPU {np.median(times) * 1000:.0f} ms/image"
    )

if args.onnx_text:
    onnx = ort.InferenceSession(args.onnx_text, so, providers=["CPUExecutionProvider"])
    sims, times = [], []
    for q in args.queries.split(","):
        tokens = txt.tokenize(q)
        t = time.perf_counter()
        a = txt.session.run(None, tokens)[0][0]
        times.append(time.perf_counter() - t)
        sims.append(cos(a, onnx.run(None, {onnx.get_inputs()[0].name: tokens["text"].astype(np.int64)})[0][0]))
    print(
        f"text  NPU vs ONNX cos: min={min(sims):.4f} mean={np.mean(sims):.4f} | "
        f"NPU {np.median(times) * 1000:.0f} ms/query"
    )
