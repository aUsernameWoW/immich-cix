#!/usr/bin/env python3
"""
CIX NPU ML server for Immich - matches the official Immich ML API format.
"""

import sys
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import time
import json
import numpy as np
from PIL import Image
import cv2
import orjson
from tokenizers import Tokenizer

from fastapi import FastAPI, File, Form, HTTPException
from fastapi.responses import ORJSONResponse, PlainTextResponse
import uvicorn

# Import CIX session
from immich_ml.sessions.cix import CixSession, is_available

app = FastAPI(title="Immich ML (CIX NPU)")

# Global sessions
clip_visual_session = None
clip_text_session = None
clip_tokenizer = None
face_det_session = None
face_rec_session = None


def load_models():
    """Load all CIX NPU models."""
    global clip_visual_session, clip_text_session, clip_tokenizer, face_det_session, face_rec_session

    cache = Path.home() / ".cache/immich_ml"

    print("Loading CIX NPU models...")

    # CLIP Visual
    path = cache / "clip/ViT-B-32__openai/visual/cix/model.cix"
    if path.exists():
        clip_visual_session = CixSession(path)
        print(f"  Loaded CLIP Visual: {path}")

    # CLIP Text
    path = cache / "clip/ViT-B-32__openai/textual/cix/model.cix"
    if path.exists():
        clip_text_session = CixSession(path)
        print(f"  Loaded CLIP Text: {path}")

    # CLIP Tokenizer
    tokenizer_path = cache / "clip/ViT-B-32__openai/textual/tokenizer.json"
    tokenizer_cfg_path = cache / "clip/ViT-B-32__openai/textual/tokenizer_config.json"
    if tokenizer_path.exists():
        clip_tokenizer = Tokenizer.from_file(str(tokenizer_path))
        # Load config for context_length and pad_token
        if tokenizer_cfg_path.exists():
            cfg = json.load(open(tokenizer_cfg_path))
            pad_token = cfg.get("pad_token", "<|endoftext|>")
        else:
            pad_token = "<|endoftext|>"
        context_length = 77  # Standard for ViT-B-32
        pad_id = clip_tokenizer.token_to_id(pad_token)
        clip_tokenizer.enable_padding(length=context_length, pad_token=pad_token, pad_id=pad_id)
        clip_tokenizer.enable_truncation(max_length=context_length)
        print(f"  Loaded CLIP Tokenizer: {tokenizer_path}")

    # Face Detection
    path = cache / "facial-recognition/buffalo_l/detection/cix/model.cix"
    if path.exists():
        face_det_session = CixSession(path)
        print(f"  Loaded Face Detection: {path}")

    # Face Recognition
    path = cache / "facial-recognition/buffalo_l/recognition/cix/model.cix"
    if path.exists():
        face_rec_session = CixSession(path)
        print(f"  Loaded Face Recognition: {path}")

    print("All models loaded!")


def preprocess_clip_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for CLIP visual encoder (OpenCLIP ViT-B-32)."""
    size = 224

    # Resize keeping aspect ratio
    w, h = image.size
    if w < h:
        new_w, new_h = size, int(h * size / w)
    else:
        new_h, new_w = size, int(w * size / h)
    image = image.resize((new_w, new_h), Image.BILINEAR)

    # Center crop
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    image = image.crop((left, top, left + size, top + size))

    # To numpy and normalize
    img_np = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    img_np = (img_np - mean) / std

    # NCHW format
    img_np = img_np.transpose(2, 0, 1)
    return np.expand_dims(img_np, 0).astype(np.float32)


def preprocess_face_detection(image: Image.Image) -> tuple[np.ndarray, tuple[int, int], float]:
    """Preprocess image for SCRFD face detection."""
    orig_w, orig_h = image.size
    target_size = 640

    # Calculate scale
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    # Resize
    image = image.resize((new_w, new_h), Image.BILINEAR)

    # Pad to 640x640
    padded = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    padded.paste(image, (0, 0))

    # Normalize
    img_np = np.array(padded, dtype=np.float32)
    img_np = (img_np - 127.5) / 128.0
    img_np = img_np.transpose(2, 0, 1)

    return np.expand_dims(img_np, 0).astype(np.float32), (orig_w, orig_h), scale


def decode_scrfd_output(outputs: list, orig_size: tuple[int, int], scale: float,
                        score_thresh: float = 0.5) -> list[dict]:
    """Decode SCRFD face detection outputs."""
    # SCRFD outputs: scores_8, scores_16, scores_32, bboxes_8, bboxes_16, bboxes_32, kps_8, kps_16, kps_32
    # For simplicity, we process the 8-stride outputs (highest resolution)

    if len(outputs) < 9:
        return []

    scores = outputs[0].flatten()  # scores_8
    bboxes = outputs[3]  # bboxes_8
    kps = outputs[6]  # kps_8

    faces = []
    for i, score in enumerate(scores):
        if score > score_thresh:
            # Get bbox (need to apply anchor decoding in real implementation)
            # For now, return simplified result
            pass

    # Return empty for now - full implementation needs anchor grid
    return []


def serialize_embedding(arr: np.ndarray) -> str:
    """Serialize numpy array to JSON string (matching Immich format)."""
    return orjson.dumps(arr.flatten().tolist()).decode()


@app.get("/")
async def root():
    return ORJSONResponse({
        "message": "Immich ML (CIX NPU)",
        "cix_available": is_available,
        "models": {
            "clip_visual": clip_visual_session is not None,
            "clip_text": clip_text_session is not None,
            "face_detection": face_det_session is not None,
            "face_recognition": face_rec_session is not None,
        }
    })


@app.get("/ping")
def ping():
    return PlainTextResponse("pong")


@app.post("/predict")
async def predict(
    entries: str = Form(...),
    image: bytes | None = File(default=None),
    text: str | None = Form(default=None),
):
    """Main prediction endpoint matching Immich ML API."""

    # Parse entries: {"clip": {"visual": {"modelName": "..."}}, "facial-recognition": {...}}
    try:
        request = orjson.loads(entries)
    except Exception as e:
        raise HTTPException(422, f"Invalid JSON: {e}")

    response = {}
    pil_image = None

    if image is not None:
        pil_image = Image.open(io.BytesIO(image)).convert("RGB")
        response["imageHeight"] = pil_image.height
        response["imageWidth"] = pil_image.width

    # Process each task
    for task, types in request.items():
        for type_name, entry in types.items():
            model_name = entry.get("modelName", "")
            options = entry.get("options", {})

            # CLIP Visual
            if task == "clip" and type_name == "visual":
                if clip_visual_session and pil_image:
                    start = time.perf_counter()
                    img_tensor = preprocess_clip_image(pil_image)
                    result = clip_visual_session.run(None, {"input.1": img_tensor})
                    embedding = result[0].flatten()
                    elapsed = (time.perf_counter() - start) * 1000
                    print(f"CLIP Visual: {elapsed:.1f}ms, shape={embedding.shape}")
                    # Return as string "[0.1, 0.2, ...]" format for Immich
                    response["clip"] = orjson.dumps(embedding.tolist()).decode()
                else:
                    raise HTTPException(500, "CLIP visual model not available")

            # CLIP Textual
            elif task == "clip" and type_name == "textual":
                if clip_text_session and clip_tokenizer and text:
                    start = time.perf_counter()
                    # Tokenize text
                    tokens = clip_tokenizer.encode(text)
                    token_ids = np.array([tokens.ids], dtype=np.int32)
                    # Run inference
                    result = clip_text_session.run(None, {"text": token_ids})
                    embedding = result[0].flatten()
                    elapsed = (time.perf_counter() - start) * 1000
                    print(f"CLIP Text: {elapsed:.1f}ms, query='{text}', shape={embedding.shape}")
                    # Return as string "[0.1, 0.2, ...]" format for Immich
                    response["clip"] = orjson.dumps(embedding.tolist()).decode()
                else:
                    raise HTTPException(500, "CLIP text model/tokenizer not available or no text provided")

            # Face Detection
            elif task == "facial-recognition" and type_name == "detection":
                if face_det_session and pil_image:
                    start = time.perf_counter()
                    img_tensor, orig_size, scale = preprocess_face_detection(pil_image)
                    outputs = face_det_session.run(None, {"input.1": img_tensor})

                    min_score = options.get("minScore", 0.7)
                    faces = decode_scrfd_output(outputs, orig_size, scale, min_score)

                    elapsed = (time.perf_counter() - start) * 1000
                    print(f"Face Detection: {elapsed:.1f}ms, found {len(faces)} faces")

                    # Return empty list for now (full implementation needs post-processing)
                    response["facial-recognition"] = []
                else:
                    response["facial-recognition"] = []

            # Face Recognition
            elif task == "facial-recognition" and type_name == "recognition":
                # This is called after detection with face crops
                response["facial-recognition"] = []

    return ORJSONResponse(response)


if __name__ == "__main__":
    print(f"CIX NPU available: {is_available}")
    load_models()
    uvicorn.run(app, host="0.0.0.0", port=3003)
