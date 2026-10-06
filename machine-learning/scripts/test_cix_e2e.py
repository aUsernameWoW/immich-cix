#!/usr/bin/env python3
"""
End-to-end test of CIX NPU integration with real images.

Tests:
1. CLIP Visual - Image embedding generation
2. CLIP Textual - Text embedding generation
3. SCRFD - Face detection
4. ArcFace - Face recognition embeddings
"""

import sys
import time
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import cv2


def load_and_preprocess_clip(image_path: str) -> np.ndarray:
    """Load and preprocess image for CLIP visual encoder."""
    # Load image
    img = Image.open(image_path).convert("RGB")

    # Resize to 224x224 with center crop
    size = 224
    # Resize keeping aspect ratio
    w, h = img.size
    if w < h:
        new_w = size
        new_h = int(h * size / w)
    else:
        new_h = size
        new_w = int(w * size / h)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    img = img.crop((left, top, left + size, top + size))

    # Convert to numpy and normalize
    img_np = np.array(img, dtype=np.float32) / 255.0

    # CLIP normalization
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    img_np = (img_np - mean) / std

    # NCHW format
    img_np = img_np.transpose(2, 0, 1)
    img_np = np.expand_dims(img_np, 0)

    return img_np


def load_and_preprocess_scrfd(image_path: str) -> tuple:
    """Load and preprocess image for SCRFD face detection."""
    img = cv2.imread(image_path)
    original_shape = img.shape[:2]

    # Resize to 640x640
    target_size = (640, 640)
    img_resized = cv2.resize(img, target_size)

    # Normalize with SCRFD params
    mean = 127.5
    std = 128.0
    blob = cv2.dnn.blobFromImage(
        img_resized,
        1.0 / std,
        target_size,
        (mean, mean, mean),
        swapRB=True
    )

    return blob, original_shape, img


def test_clip_visual(image_path: str):
    """Test CLIP visual encoder with CIX NPU."""
    print("\n" + "=" * 60)
    print("Test: CLIP Visual Encoder")
    print("=" * 60)
    print(f"  Image: {image_path}")

    from immich_ml.sessions.cix import CixSession

    model_path = Path.home() / ".cache/immich_ml/clip/ViT-B-32__openai/visual/cix/model.cix"
    if not model_path.exists():
        print(f"  [SKIP] Model not found: {model_path}")
        return None

    # Load model
    print("  Loading model...")
    session = CixSession(model_path)

    # Preprocess image
    print("  Preprocessing image...")
    img_tensor = load_and_preprocess_clip(image_path)
    print(f"  Input shape: {img_tensor.shape}")

    # Run inference
    print("  Running inference...")
    start = time.perf_counter()
    result = session.run(None, {"input.1": img_tensor})
    elapsed = (time.perf_counter() - start) * 1000

    embedding = result[0]
    print(f"  [OK] Inference completed in {elapsed:.2f} ms")
    print(f"  Output shape: {embedding.shape}")
    print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
    print(f"  Sample values: {embedding[:5]}")

    session.release()
    return embedding


def test_clip_textual(query: str):
    """Test CLIP text encoder with CIX NPU."""
    print("\n" + "=" * 60)
    print("Test: CLIP Textual Encoder")
    print("=" * 60)
    print(f"  Query: '{query}'")

    try:
        import clip
    except ImportError:
        print("  [SKIP] 'clip' package not installed")
        return None

    from immich_ml.sessions.cix import CixSession

    model_path = Path.home() / ".cache/immich_ml/clip/ViT-B-32__openai/textual/cix/model.cix"
    if not model_path.exists():
        print(f"  [SKIP] Model not found: {model_path}")
        return None

    # Load model
    print("  Loading model...")
    session = CixSession(model_path)

    # Tokenize text
    print("  Tokenizing text...")
    tokens = clip.tokenize(query).numpy().astype(np.int32)
    print(f"  Token shape: {tokens.shape}")

    # Run inference
    print("  Running inference...")
    start = time.perf_counter()
    result = session.run(None, {"text": tokens})
    elapsed = (time.perf_counter() - start) * 1000

    embedding = result[0]
    print(f"  [OK] Inference completed in {elapsed:.2f} ms")
    print(f"  Output shape: {embedding.shape}")
    print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")

    session.release()
    return embedding


def test_face_detection(image_path: str):
    """Test SCRFD face detection with CIX NPU."""
    print("\n" + "=" * 60)
    print("Test: SCRFD Face Detection")
    print("=" * 60)
    print(f"  Image: {image_path}")

    from immich_ml.sessions.cix import CixSession

    model_path = Path.home() / ".cache/immich_ml/facial-recognition/buffalo_l/detection/cix/model.cix"
    if not model_path.exists():
        print(f"  [SKIP] Model not found: {model_path}")
        return None

    # Load model
    print("  Loading model...")
    session = CixSession(model_path)

    # Preprocess image
    print("  Preprocessing image...")
    blob, original_shape, original_img = load_and_preprocess_scrfd(image_path)
    print(f"  Input shape: {blob.shape}")
    print(f"  Original size: {original_shape}")

    # Run inference
    print("  Running inference...")
    start = time.perf_counter()
    outputs = session.run(None, {"input.1": blob})
    elapsed = (time.perf_counter() - start) * 1000

    print(f"  [OK] Inference completed in {elapsed:.2f} ms")
    print(f"  Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"    Output {i}: shape={out.shape}")

    session.release()
    return outputs, original_img, original_shape


def test_face_recognition(image_path: str):
    """Test ArcFace face recognition with CIX NPU."""
    print("\n" + "=" * 60)
    print("Test: ArcFace Face Recognition")
    print("=" * 60)
    print(f"  Image: {image_path}")

    from immich_ml.sessions.cix import CixSession

    model_path = Path.home() / ".cache/immich_ml/facial-recognition/buffalo_l/recognition/cix/model.cix"
    if not model_path.exists():
        print(f"  [SKIP] Model not found: {model_path}")
        return None

    # Load model
    print("  Loading model...")
    session = CixSession(model_path)

    # For ArcFace, we need a cropped face image (112x112)
    # Here we just use a dummy aligned face for testing
    print("  Creating test input (112x112 face crop)...")

    # Load and create a 112x112 crop from center of image
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Center crop to square then resize to 112x112
    min_dim = min(h, w)
    start_h = (h - min_dim) // 2
    start_w = (w - min_dim) // 2
    crop = img[start_h:start_h+min_dim, start_w:start_w+min_dim]
    face_crop = cv2.resize(crop, (112, 112))

    # Normalize for ArcFace
    mean = 127.5
    std = 127.5
    blob = cv2.dnn.blobFromImage(
        face_crop,
        1.0 / std,
        (112, 112),
        (mean, mean, mean),
        swapRB=True
    )
    print(f"  Input shape: {blob.shape}")

    # Run inference
    print("  Running inference...")
    start = time.perf_counter()
    result = session.run(None, {"input.1": blob})
    elapsed = (time.perf_counter() - start) * 1000

    embedding = result[0]
    print(f"  [OK] Inference completed in {elapsed:.2f} ms")
    print(f"  Output shape: {embedding.shape}")
    print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")

    session.release()
    return embedding


def test_similarity(img_embedding, text_embedding):
    """Test similarity computation between image and text embeddings."""
    print("\n" + "=" * 60)
    print("Test: CLIP Similarity")
    print("=" * 60)

    if img_embedding is None or text_embedding is None:
        print("  [SKIP] Missing embeddings")
        return

    # Normalize embeddings
    img_norm = img_embedding / np.linalg.norm(img_embedding)
    text_norm = text_embedding / np.linalg.norm(text_embedding)

    # Compute cosine similarity
    similarity = np.dot(img_norm, text_norm)
    print(f"  Cosine similarity: {similarity:.4f}")
    print(f"  (Higher = more similar, range: -1 to 1)")


def main():
    print("\n" + "=" * 60)
    print("CIX NPU End-to-End Test with Real Images")
    print("=" * 60)

    # Find test images
    test_dir = Path("/mnt/tank/media/photos")
    test_images = list(test_dir.glob("**/*.JPG"))[:3]

    if not test_images:
        print("ERROR: No test images found in /mnt/tank/media/photos")
        return False

    print(f"\nFound {len(test_images)} test images")

    results = {}

    # Test with first image
    test_image = str(test_images[0])
    print(f"\nUsing test image: {test_image}")

    # Test CLIP Visual
    try:
        img_embedding = test_clip_visual(test_image)
        results["CLIP Visual"] = img_embedding is not None
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        results["CLIP Visual"] = False

    # Test CLIP Textual
    try:
        text_embedding = test_clip_textual("a photo of a person")
        results["CLIP Textual"] = text_embedding is not None
    except Exception as e:
        print(f"  [FAIL] {e}")
        results["CLIP Textual"] = False

    # Test similarity if both worked
    if results.get("CLIP Visual") and results.get("CLIP Textual"):
        test_similarity(img_embedding, text_embedding)

    # Test Face Detection
    try:
        detection_result = test_face_detection(test_image)
        results["Face Detection"] = detection_result is not None
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        results["Face Detection"] = False

    # Test Face Recognition
    try:
        face_embedding = test_face_recognition(test_image)
        results["Face Recognition"] = face_embedding is not None
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        results["Face Recognition"] = False

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    passed = sum(results.values())
    total = len(results)
    print(f"\nResult: {passed}/{total} tests passed")

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
