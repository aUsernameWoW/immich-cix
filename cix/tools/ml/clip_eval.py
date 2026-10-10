"""Compare CLIP models on a sample of the real library, the way it matters for search: rank the same images for the
same queries with each model and render contact sheets (one row of top results per model).

  sample  DIR [--n 1500]                   pick random previews; stores the production embeddings as the baseline
  embed   DIR MODEL [--onnx-image PATH]    embed the sample with MODEL on the NPU (and optionally fp32 ONNX)
  query   DIR --models M1,M2 [--baseline NAME] [--onnx-text M1=PATH] --queries "q1,q2"
          -> DIR/qNN.jpg contact sheets; with ONNX rows also prints NPU-vs-ONNX top-10 overlap

Run from immich-v3/machine-learning like clip_npu_vs_onnx.py. The baseline text encoder is loaded from
~/.cache/immich_ml/clip/<baseline> (the production cache).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _db import psql  # noqa: E402

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("sample")
p.add_argument("dir")
p.add_argument("--n", type=int, default=1500)
p = sub.add_parser("embed")
p.add_argument("dir")
p.add_argument("model")
p.add_argument("--onnx-image")
p = sub.add_parser("query")
p.add_argument("dir")
p.add_argument("--models", required=True)
p.add_argument("--baseline", default="chinese-clip-vit-large-patch14", help="model that produced the stored embeddings")
p.add_argument("--onnx-text", action="append", default=[], help="MODEL=path of the fp32 text encoder")
p.add_argument("--queries", required=True)
args = parser.parse_args()
d = Path(args.dir)


def norm(m):
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


if args.cmd == "sample":
    d.mkdir(parents=True, exist_ok=True)
    rows = psql(f"""select a.id, f.path, s.embedding::text from asset a
                    join asset_file f on f."assetId"=a.id and f.type='preview' and f."isEdited"=false
                    join smart_search s on s."assetId"=a.id
                    where a.type='IMAGE' and a."deletedAt" is null and a.visibility='timeline'
                    order by random() limit {args.n}""")
    (d / "sample.psv").write_text("\n".join("|".join(r) for r in rows))
    print(f"sampled {len(rows)} images")

elif args.cmd == "embed":
    import onnxruntime as ort
    from immich_ml.models.clip.visual import OpenClipVisualEncoder
    from immich_ml.models.transforms import decode_pil

    rows = [line.split("|", 2) for line in (d / "sample.psv").read_text().splitlines()]
    vis = OpenClipVisualEncoder(args.model)
    vis.load()
    onnx = None
    if args.onnx_image:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        onnx = ort.InferenceSession(args.onnx_image, so, providers=["CPUExecutionProvider"])
    npu, ref = [], []
    for _, path, _ in rows:
        x = vis.transform(decode_pil(open(path, "rb").read()))["image"]
        npu.append(vis.session.run(None, {"image": x})[0][0])
        if onnx:
            ref.append(onnx.run(None, {onnx.get_inputs()[0].name: x})[0][0])
    np.savez(d / f"{args.model}.npz", npu=np.array(npu), onnx=np.array(ref))
    print(f"{args.model}: embedded {len(rows)} images")

else:
    import onnxruntime as ort
    from immich_ml.models.clip.textual import OpenClipTextualEncoder
    from PIL import Image, ImageDraw, ImageFont

    rows = [line.split("|", 2) for line in (d / "sample.psv").read_text().splitlines()]
    paths = [r[1] for r in rows]
    so = ort.SessionOptions()
    so.log_severity_level = 3
    baseline = OpenClipTextualEncoder(args.baseline, cache_dir=Path.home() / ".cache/immich_ml/clip" / args.baseline)
    baseline.load()
    variants = {
        f"{args.baseline} (stored)": (
            norm(np.array([json.loads(r[2]) for r in rows], dtype=np.float32)),
            baseline,
            None,
        )
    }
    onnx_text = dict(item.split("=", 1) for item in args.onnx_text)
    for model in args.models.split(","):
        emb = np.load(d / f"{model}.npz")
        enc = OpenClipTextualEncoder(model)
        enc.load()
        variants[f"{model} NPU"] = (norm(emb["npu"]), enc, None)
        if model in onnx_text and emb["onnx"].size:
            variants[f"{model} ONNX"] = (
                norm(emb["onnx"]),
                enc,
                ort.InferenceSession(onnx_text[model], so, providers=["CPUExecutionProvider"]),
            )

    def text_vec(enc, onnx, q):
        tokens = enc.tokenize(q)
        if onnx is None:
            return norm(enc.session.run(None, tokens)[0][0])
        return norm(onnx.run(None, {onnx.get_inputs()[0].name: tokens["text"].astype(np.int64)})[0][0])

    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font = ImageFont.truetype(font_path, 16) if Path(font_path).exists() else ImageFont.load_default()
    size, k, label_w = 150, 5, 260
    for qi, q in enumerate(args.queries.split(",")):
        sheet = Image.new("RGB", (label_w + k * size, 30 + len(variants) * size), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((5, 5), f"query: {q}", fill="black", font=font)
        tops = {}
        for row, (name, (emb, enc, onnx)) in enumerate(variants.items()):
            top = np.argsort(-(emb @ text_vec(enc, onnx, q)))[:10]
            tops[name] = top
            draw.text((5, 30 + row * size + size // 2 - 8), name, fill="black", font=font)
            for col, idx in enumerate(top[:k]):
                im = Image.open(paths[idx]).convert("RGB")
                im.thumbnail((size - 4, size - 4))
                sheet.paste(im, (label_w + col * size + 2, 30 + row * size + 2))
        sheet.save(d / f"q{qi:02d}.jpg", quality=80)
        overlaps = [
            f"{m}: {len(set(tops[f'{m} NPU']) & set(tops[f'{m} ONNX'])) / 10:.1f}"
            for m in args.models.split(",")
            if f"{m} ONNX" in tops
        ]
        print(f"q{qi:02d} {q}" + (f"  NPU/ONNX top-10 overlap {', '.join(overlaps)}" if overlaps else ""))
