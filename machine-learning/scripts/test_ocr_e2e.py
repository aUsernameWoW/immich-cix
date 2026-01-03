#!/usr/bin/env python3
"""
End-to-end test for CIX NPU OCR via HTTP API.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import json
from PIL import Image
import io

# Test configuration
ML_SERVER_URL = "http://localhost:3003"
TEST_IMAGE_DIR = Path.home() / ".cache/modelscope/hub/models/cix/ai_model_hub_25_Q3/models/ComputeVision/OCR/onnx_PP_OCRv4/test_data"


def test_ocr_api(image_path: Path):
    """Test OCR via ML server API."""
    print(f"\n=== Testing OCR API with {image_path.name} ===")

    # Load image
    with open(image_path, "rb") as f:
        image_data = f.read()

    # Prepare request
    entries = json.dumps({
        "ocr": {
            "detection": {
                "modelName": "PP-OCRv4_mobile",
                "options": {"minScore": 0.5}
            },
            "recognition": {
                "modelName": "PP-OCRv4_mobile",
                "options": {"minScore": 0.5}
            }
        }
    })

    files = {
        "image": ("image.jpg", image_data, "image/jpeg"),
    }
    data = {
        "entries": entries,
    }

    try:
        response = requests.post(f"{ML_SERVER_URL}/predict", files=files, data=data, timeout=60)
        response.raise_for_status()

        result = response.json()

        print(f"Image size: {result.get('imageWidth')}x{result.get('imageHeight')}")

        if "ocr" in result:
            ocr = result["ocr"]
            texts = ocr.get("text", [])
            boxes = ocr.get("box", [])
            box_scores = ocr.get("boxScore", [])
            text_scores = ocr.get("textScore", [])

            print(f"Detected {len(texts)} text regions:")
            for i, text in enumerate(texts):
                score = text_scores[i] if i < len(text_scores) else 0
                print(f"  [{i+1}] ({score:.2f}) {text}")
        else:
            print("No OCR result in response")
            print(f"Response: {result}")

    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to ML server at {ML_SERVER_URL}")
        print("Please start the server first: python3 scripts/run_cix_ml_server.py")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

    return True


def main():
    # Check server health
    print("Checking ML server...")
    try:
        r = requests.get(f"{ML_SERVER_URL}/", timeout=5)
        data = r.json()
        print(f"Server: {data.get('message')}")
        print(f"Models: {data.get('models')}")
    except Exception as e:
        print(f"Cannot connect to server: {e}")
        print(f"\nPlease start the server first:")
        print(f"  cd /home/radxa/immich-cix/immich-cix/machine-learning")
        print(f"  python3 scripts/run_cix_ml_server.py")
        return

    # Find test images
    test_images = list(TEST_IMAGE_DIR.glob("*.jpg")) + list(TEST_IMAGE_DIR.glob("*.png"))
    print(f"\nFound {len(test_images)} test images")

    # Test with each image
    for img_path in test_images[:3]:  # Test first 3
        test_ocr_api(img_path)

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
