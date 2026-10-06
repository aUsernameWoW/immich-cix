#!/usr/bin/env python3
"""
Test CIX NPU Session integration for Immich ML.

This script verifies that:
1. CIX NPU is available (libnoe, /dev/aipu)
2. CixSession can load .cix models
3. Inference works correctly
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def test_npu_availability():
    """Test if CIX NPU is available."""
    print("=" * 60)
    print("Test 1: CIX NPU Availability")
    print("=" * 60)

    # Check /dev/aipu
    if os.path.exists("/dev/aipu"):
        print("  [OK] /dev/aipu exists")
    else:
        print("  [WARN] /dev/aipu not found - NPU driver may not be loaded")

    # Check libnoe
    try:
        from libnoe import NPU
        print("  [OK] libnoe imported successfully")

        npu = NPU()
        if npu.noe_init_context() == 0:
            print("  [OK] NPU context initialized")
            npu.noe_deinit_context()
            return True
        else:
            print("  [FAIL] Failed to initialize NPU context")
            return False
    except ImportError as e:
        print(f"  [FAIL] Failed to import libnoe: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] NPU init error: {e}")
        return False


def test_cix_session_import():
    """Test if CixSession can be imported."""
    print("\n" + "=" * 60)
    print("Test 2: CixSession Import")
    print("=" * 60)

    try:
        from immich_ml.sessions.cix import CixSession, is_available, model_prefix
        print(f"  [OK] CixSession imported")
        print(f"  [INFO] is_available = {is_available}")
        print(f"  [INFO] model_prefix = {model_prefix}")
        return True
    except Exception as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_model_loading():
    """Test loading CIX models."""
    print("\n" + "=" * 60)
    print("Test 3: Model Loading")
    print("=" * 60)

    cache_dir = Path.home() / ".cache" / "immich_ml"
    models = {
        "CLIP Visual": cache_dir / "clip/ViT-B-32__openai/visual/cix/model.cix",
        "CLIP Text": cache_dir / "clip/ViT-B-32__openai/textual/cix/model.cix",
        "SCRFD": cache_dir / "facial-recognition/buffalo_l/detection/cix/model.cix",
        "ArcFace": cache_dir / "facial-recognition/buffalo_l/recognition/cix/model.cix",
    }

    all_exist = True
    for name, path in models.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {name}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"  [FAIL] {name}: {path} NOT FOUND")
            all_exist = False

    return all_exist


def test_inference():
    """Test inference with CIX NPU."""
    print("\n" + "=" * 60)
    print("Test 4: Inference Test")
    print("=" * 60)

    try:
        from immich_ml.sessions.cix import CixSession

        cache_dir = Path.home() / ".cache" / "immich_ml"
        visual_model = cache_dir / "clip/ViT-B-32__openai/visual/cix/model.cix"

        if not visual_model.exists():
            print(f"  [SKIP] Model not found: {visual_model}")
            return False

        print(f"  Loading model: {visual_model}")
        session = CixSession(visual_model)

        # Get input/output info
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        print(f"  [OK] Model loaded")
        print(f"       Inputs: {[(i.name, i.shape) for i in inputs]}")
        print(f"       Outputs: {[(o.name, o.shape) for o in outputs]}")

        # Create dummy input (1, 3, 224, 224)
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

        print(f"  Running inference...")
        import time
        start = time.perf_counter()
        result = session.run(None, {"input.1": dummy_input})
        elapsed = (time.perf_counter() - start) * 1000

        print(f"  [OK] Inference completed in {elapsed:.2f} ms")
        print(f"       Output shape: {result[0].shape}")
        print(f"       Output sample: {result[0][:5]}...")

        # Cleanup
        session.release()
        print(f"  [OK] Session released")

        return True

    except Exception as e:
        import traceback
        print(f"  [FAIL] Inference error: {e}")
        traceback.print_exc()
        return False


def test_base_model_integration():
    """Test integration with Immich base model."""
    print("\n" + "=" * 60)
    print("Test 5: Base Model Integration")
    print("=" * 60)

    try:
        from immich_ml.schemas import ModelFormat
        from immich_ml.sessions import cix

        print(f"  [OK] ModelFormat.CIX = {ModelFormat.CIX}")
        print(f"  [INFO] cix.is_available = {cix.is_available}")

        # Check if CIX is prioritized
        if cix.is_available:
            print(f"  [OK] CIX NPU will be used by default")
        else:
            print(f"  [INFO] CIX NPU not available, will fall back to other backends")

        return True

    except Exception as e:
        print(f"  [FAIL] Integration error: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("CIX NPU Integration Test for Immich ML")
    print("=" * 60 + "\n")

    results = {}

    # Run tests
    results["NPU Availability"] = test_npu_availability()
    results["CixSession Import"] = test_cix_session_import()
    results["Model Loading"] = test_model_loading()

    # Only run inference if NPU is available
    if results["NPU Availability"] and results["Model Loading"]:
        results["Inference"] = test_inference()
    else:
        print("\n[SKIP] Inference test - NPU or models not available")
        results["Inference"] = None

    results["Base Model Integration"] = test_base_model_integration()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, result in results.items():
        if result is True:
            status = "[PASS]"
        elif result is False:
            status = "[FAIL]"
        else:
            status = "[SKIP]"
        print(f"  {status} {name}")

    passed = sum(1 for r in results.values() if r is True)
    total = sum(1 for r in results.values() if r is not None)
    print(f"\nResult: {passed}/{total} tests passed")

    return all(r is not False for r in results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
