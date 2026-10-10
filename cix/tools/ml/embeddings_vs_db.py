"""Recompute embeddings with an immich_ml server and compare them with the ones stored in the database.

  clip  : smart_search embeddings (should be ~1.0 when the model and pipeline are unchanged)
  faces : face_search embeddings, matched to the stored box by IoU (Immich clusters people at cosine distance
          0.5, so anything well above 0.5 keeps existing people intact)

Usage: embeddings_vs_db.py clip|faces http://127.0.0.1:3003 [--limit 40] [--random] [--model NAME]
Reads the DB through `podman exec $PG_CONTAINER psql` (see _db.py).
"""

import argparse
import json

import numpy as np
import requests
from _db import psql

parser = argparse.ArgumentParser()
parser.add_argument("kind", choices=["clip", "faces"])
parser.add_argument("url")
parser.add_argument("--limit", type=int, default=40)
parser.add_argument(
    "--random", action="store_true", help="random sample instead of a fixed one (fixed is comparable across runs)"
)
parser.add_argument("--model")
args = parser.parse_args()
order = "random()" if args.random else "1"
preview = """join asset_file f on f."assetId"={id} and f.type='preview' and f."isEdited"=false"""


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def vec(x):
    return np.asarray(json.loads(x), dtype=np.float32)


sims = []

if args.kind == "clip":
    model = args.model or "ViT-B-32__openai"
    rows = psql(f"""select f.path, s.embedding::text from smart_search s {preview.format(id='s."assetId"')}
                    order by {order} limit {args.limit}""")
    entries = {"clip": {"visual": {"modelName": model, "options": {}}}}
    for path, emb in rows:
        r = requests.post(
            f"{args.url}/predict", data={"entries": json.dumps(entries)}, files={"image": open(path, "rb")}, timeout=120
        )
        sims.append(cos(vec(r.json()["clip"]), vec(emb)))
else:
    model = args.model or "buffalo_l"
    rows = psql(f"""select f.path, af."boundingBoxX1", af."boundingBoxY1", af."boundingBoxX2", af."boundingBoxY2",
                           fs.embedding::text
                    from asset_face af join face_search fs on fs."faceId"=af.id {preview.format(id='af."assetId"')}
                    where af."deletedAt" is null and af."sourceType"='machine-learning'
                    order by {order if args.random else "af.id"} limit {args.limit}""")
    entries = {
        "facial-recognition": {
            "detection": {"modelName": model, "options": {"minScore": 0.5}},
            "recognition": {"modelName": model},
        }
    }
    for path, x1, y1, x2, y2, emb in rows:
        stored = np.array([x1, y1, x2, y2], dtype=np.float32)
        r = requests.post(
            f"{args.url}/predict", data={"entries": json.dumps(entries)}, files={"image": open(path, "rb")}, timeout=120
        )
        best = None
        for g in r.json()["facial-recognition"]:
            b = np.array([g["boundingBox"][k] for k in ("x1", "y1", "x2", "y2")], dtype=np.float32)
            ix = max(0, min(stored[2], b[2]) - max(stored[0], b[0]))
            iy = max(0, min(stored[3], b[3]) - max(stored[1], b[1]))
            iou = (
                ix * iy / ((stored[2] - stored[0]) * (stored[3] - stored[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy)
            )
            if best is None or iou > best[0]:
                best = (iou, cos(vec(g["embedding"]), vec(emb)))
        if best and best[0] > 0.5:
            sims.append(best[1])

s = np.array(sims)
print(
    f"{args.kind} {model} @ {args.url}: {len(s)}/{len(rows)} compared  "
    f"cos vs DB: min={s.min():.4f} p10={np.percentile(s, 10):.4f} median={np.median(s):.4f}"
)
