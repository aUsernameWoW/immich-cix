"""Send the same /predict requests to two immich_ml servers and compare CLIP and face results.

Usage: compare_servers.py http://127.0.0.1:3003 http://127.0.0.1:3013 img1.jpg img2.jpg ...
       (use preview JPEGs from the thumbs folder: that's what the server sends to ML)
"""

import argparse
import json
import time

import numpy as np
import requests

parser = argparse.ArgumentParser()
parser.add_argument("a")
parser.add_argument("b")
parser.add_argument("images", nargs="+")
parser.add_argument("--clip", default="ViT-B-32__openai")
parser.add_argument("--faces", default="buffalo_l")
parser.add_argument("--texts", default="a dog on the beach,生日蛋糕")
args = parser.parse_args()


def call(url, entries, image=None, text=None):
    data = {"entries": json.dumps(entries)}
    if text:
        data["text"] = text
    files = {"image": open(image, "rb")} if image else None
    t = time.time()
    r = requests.post(f"{url}/predict", data=data, files=files, timeout=120)
    r.raise_for_status()
    return r.json(), (time.time() - t) * 1000


def vec(x):
    return np.asarray(json.loads(x) if isinstance(x, str) else x, dtype=np.float32)


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0]))
    iy = max(0, min(p[3], q[3]) - max(p[1], q[1]))
    inter = ix * iy
    return inter / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter)


def box(f):
    return [f["boundingBox"][k] for k in ("x1", "y1", "x2", "y2")]


textual = {"clip": {"textual": {"modelName": args.clip, "options": {}}}}
visual = {"clip": {"visual": {"modelName": args.clip, "options": {}}}}
faces = {
    "facial-recognition": {
        "detection": {"modelName": args.faces, "options": {"minScore": 0.7}},
        "recognition": {"modelName": args.faces},
    }
}

for text in args.texts.split(","):
    (ra, ta), (rb, tb) = call(args.a, textual, text=text), call(args.b, textual, text=text)
    print(f"[clip-text] {text!r}: cos={cos(vec(ra['clip']), vec(rb['clip'])):.5f}  A {ta:.0f}ms B {tb:.0f}ms")

for img in args.images:
    name = img.rsplit("/", 1)[-1][:16]
    (ra, ta), (rb, tb) = call(args.a, visual, image=img), call(args.b, visual, image=img)
    print(f"[clip-img ] {name}: cos={cos(vec(ra['clip']), vec(rb['clip'])):.5f}  A {ta:.0f}ms B {tb:.0f}ms")
    (ra, ta), (rb, tb) = call(args.a, faces, image=img), call(args.b, faces, image=img)
    fa, fb = ra["facial-recognition"], rb["facial-recognition"]
    print(f"[faces    ] {name}: A {len(fa)} faces {ta:.0f}ms | B {len(fb)} faces {tb:.0f}ms")
    for f in fa:
        best = max(
            ((iou(box(f), box(g)), cos(vec(f["embedding"]), vec(g["embedding"])), g["score"]) for g in fb), default=None
        )
        if best:
            print(f"    score A={f['score']:.3f} B={best[2]:.3f} IoU={best[0]:.3f} embedding cos={best[1]:.4f}")
        else:
            print(f"    score A={f['score']:.3f}: no match in B")
