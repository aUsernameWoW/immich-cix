#!/usr/bin/env python3
"""
Test script for CIX NPU OCR functionality.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import cv2
import time

from immich_ml.sessions.cix import CixSession, is_available

# Test image path - use a test image with text
TEST_IMAGE = Path.home() / ".cache/modelscope/hub/models/cix/ai_model_hub_25_Q3/models/ComputeVision/OCR/onnx_PP_OCRv4/test_data"


def main():
    print(f"CIX NPU available: {is_available}")

    cache = Path.home() / ".cache/immich_ml"

    # Load models
    print("\n=== Loading Models ===")

    det_path = cache / "ocr/PP-OCRv4_mobile/detection/cix/model.cix"
    rec_path = cache / "ocr/PP-OCRv4_mobile/recognition/cix/model.cix"
    dict_path = cache / "ocr/PP-OCRv4_mobile/ppocr_keys_v1.txt"

    if not det_path.exists():
        print(f"Detection model not found: {det_path}")
        return
    if not rec_path.exists():
        print(f"Recognition model not found: {rec_path}")
        return

    det_session = CixSession(det_path)
    print(f"Detection model type: {det_session.model_type}")
    print(f"Detection inputs: {det_session.get_inputs()}")
    print(f"Detection outputs: {det_session.get_outputs()}")

    rec_session = CixSession(rec_path)
    print(f"Recognition model type: {rec_session.model_type}")
    print(f"Recognition inputs: {rec_session.get_inputs()}")
    print(f"Recognition outputs: {rec_session.get_outputs()}")

    # Load character dictionary
    with open(dict_path, 'r', encoding='utf-8') as f:
        char_dict = [''] + [line.strip() for line in f.readlines()]
        char_dict.append(' ')
    print(f"Character dictionary: {len(char_dict)} characters")

    # Find test image
    test_images = list(TEST_IMAGE.glob("*.jpg")) + list(TEST_IMAGE.glob("*.png"))
    if not test_images:
        print(f"No test images found in {TEST_IMAGE}")
        # Create a simple test image with text
        print("Creating synthetic test image...")
        img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img, "Hello World!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        cv2.putText(img, "OCR Test", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        test_img_path = Path("/tmp/ocr_test.png")
        cv2.imwrite(str(test_img_path), img)
        test_images = [test_img_path]

    print(f"\n=== Testing with {len(test_images)} images ===")

    for img_path in test_images[:2]:  # Test first 2 images
        print(f"\nProcessing: {img_path}")

        # Load image
        image = Image.open(img_path).convert("RGB")
        print(f"Image size: {image.size}")

        # Preprocess for detection
        start = time.perf_counter()

        # Target size for CIX model: 960x608
        target_h, target_w = 960, 608
        orig_w, orig_h = image.size

        # Resize keeping aspect ratio
        ratio = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * ratio)
        new_h = int(orig_h * ratio)
        new_w = max(int(round(new_w / 32) * 32), 32)
        new_h = max(int(round(new_h / 32) * 32), 32)

        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_np = np.array(resized, dtype=np.float32)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
        img_np = (img_np - mean) / std

        # Pad to target size
        padded = np.zeros((target_h, target_w, 3), dtype=np.float32)
        padded[:new_h, :new_w, :] = img_np

        # NCHW format
        det_input = padded.transpose(2, 0, 1)
        det_input = np.expand_dims(det_input, 0).astype(np.float32)

        preprocess_time = (time.perf_counter() - start) * 1000
        print(f"Preprocessing: {preprocess_time:.1f}ms")
        print(f"Detection input shape: {det_input.shape}")

        # Run detection
        start = time.perf_counter()
        try:
            det_output = det_session.run(None, {"x": det_input})
            det_time = (time.perf_counter() - start) * 1000
            print(f"Detection inference: {det_time:.1f}ms")
            print(f"Detection output shape: {det_output[0].shape}")
            print(f"Detection output range: [{det_output[0].min():.4f}, {det_output[0].max():.4f}]")

            # Check if we have text regions
            det_pred = det_output[0].reshape(1, 1, target_h, target_w)
            thresh = 0.3
            text_mask = det_pred[0, 0] > thresh
            num_text_pixels = np.sum(text_mask)
            print(f"Text pixels (>{thresh}): {num_text_pixels}")

        except Exception as e:
            print(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Simple test of recognition model
        print("\nTesting recognition model with dummy input...")
        rec_input = np.random.randn(1, 3, 32, 400).astype(np.float32)
        try:
            start = time.perf_counter()
            rec_output = rec_session.run(None, {"x": rec_input})
            rec_time = (time.perf_counter() - start) * 1000
            print(f"Recognition inference: {rec_time:.1f}ms")
            print(f"Recognition output shape: {rec_output[0].shape}")
        except Exception as e:
            print(f"Recognition error: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
