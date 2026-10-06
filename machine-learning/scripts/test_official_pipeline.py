#!/usr/bin/env python3
"""
Test compatibility of CIX NPU sessions with the official Immich ML pipeline.

This script tests whether each model type (CLIP visual/textual, face detection,
face recognition) can work through the official Immich pipeline classes with
CixSession, identifying blockers for deprecating run_cix_ml_server.py.

Usage:
    PYTHONPATH=/usr/local/lib/python3.11/dist-packages \
    LD_LIBRARY_PATH=/usr/share/cix/lib \
    python3 scripts/test_official_pipeline.py
"""

import sys
import os
import traceback
from pathlib import Path

# Ensure we can find the immich_ml package
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

CACHE = Path.home() / ".cache" / "immich_ml"
RESULTS: dict[str, str] = {}


def report(name: str, status: str, detail: str = "") -> None:
    tag = "✅ PASS" if status == "pass" else "❌ FAIL" if status == "fail" else "⚠️  SKIP"
    msg = f"{tag}  {name}"
    if detail:
        msg += f"  — {detail}"
    print(msg)
    RESULTS[name] = status


# ──────────────────────────────────────────────────────────────
# 1. Test CixSession directly (sanity check)
# ──────────────────────────────────────────────────────────────

def test_cix_session_basic():
    """Verify CixSession loads and exposes get_inputs/get_outputs correctly."""
    from immich_ml.sessions.cix import CixSession, is_available

    if not is_available:
        report("CixSession.is_available", "fail", "CIX NPU not available on this system")
        return False

    report("CixSession.is_available", "pass")

    # Test detection model metadata
    det_path = CACHE / "facial-recognition/buffalo_l/detection/cix/model.cix"
    if not det_path.exists():
        report("CixSession.detection_metadata", "skip", f"Model not found: {det_path}")
        return False

    sess = CixSession(det_path)
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()

    print(f"  Detection model: {len(inputs)} inputs, {len(outputs)} outputs")
    print(f"  Input[0]: name={inputs[0].name}, shape={inputs[0].shape}")
    for i, o in enumerate(outputs):
        print(f"  Output[{i}]: name={o.name}, shape={o.shape}")

    # Verify shapes are tuples of ints (not strings)
    assert isinstance(inputs[0].shape[2], int), "Input shape should have int dims"
    assert len(outputs) == 9, f"Expected 9 outputs for SCRFD, got {len(outputs)}"

    report("CixSession.detection_metadata", "pass", "9 outputs, shapes are int tuples")
    sess.release()
    return True


# ──────────────────────────────────────────────────────────────
# 2. Test RetinaFace compatibility
# ──────────────────────────────────────────────────────────────

def test_retinaface_init():
    """Test if RetinaFace.__init__ + _init_vars works with CixSession."""
    from insightface.model_zoo.retinaface import RetinaFace
    from immich_ml.sessions.cix import CixSession

    det_path = CACHE / "facial-recognition/buffalo_l/detection/cix/model.cix"
    if not det_path.exists():
        report("RetinaFace._init_vars", "skip", "Detection model not found")
        return None

    sess = CixSession(det_path)
    try:
        model = RetinaFace(session=sess)
        model.prepare(ctx_id=0, det_thresh=0.7, input_size=(640, 640))
        print(f"  RetinaFace: input_size={model.input_size}, use_kps={model.use_kps}")
        print(f"  fmc={model.fmc}, strides={model._feat_stride_fpn}, num_anchors={model._num_anchors}")
        print(f"  output_names={model.output_names}")

        assert model.use_kps is True, "Expected use_kps=True for 9-output SCRFD"
        assert model.fmc == 3, f"Expected fmc=3, got {model.fmc}"
        assert model._num_anchors == 2, f"Expected 2 anchors, got {model._num_anchors}"
        assert model.input_size == (640, 640), f"Expected input_size=(640,640), got {model.input_size}"

        report("RetinaFace._init_vars", "pass", "Correctly detected 9-output SCRFD with kps")
        return (model, sess)
    except Exception as e:
        report("RetinaFace._init_vars", "fail", str(e))
        traceback.print_exc()
        sess.release()
        return None


def test_retinaface_inference():
    """Test actual face detection inference through RetinaFace + CixSession."""
    import cv2

    result = test_retinaface_init()
    if result is None:
        report("RetinaFace.forward", "skip", "Init failed")
        return

    model, sess = result

    # Create a dummy 640x640 test image with a face-like oval
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    # Draw an ellipse to simulate a face
    cv2.ellipse(img, (320, 300), (80, 100), 0, 0, 360, (200, 180, 170), -1)
    # Eyes
    cv2.circle(img, (290, 270), 10, (50, 50, 50), -1)
    cv2.circle(img, (350, 270), 10, (50, 50, 50), -1)

    try:
        # Test forward() which calls session.run()
        scores_list, bboxes_list, kpss_list = model.forward(img, 0.3)

        print(f"  Forward results: {len(scores_list)} stride groups")
        for i, (s, b, k) in enumerate(zip(scores_list, bboxes_list, kpss_list)):
            print(f"    Stride {model._feat_stride_fpn[i]}: scores={s.shape}, bboxes={b.shape}, kps={k.shape}")

        # Check output shapes — the key issue is whether outputs come back shaped correctly
        # RetinaFace expects: net_outs[idx] = scores with shape that can be indexed by [pos_inds]
        # and net_outs[idx+fmc] = bbox_preds, etc.
        report("RetinaFace.forward", "pass", "Inference completed without error")

    except Exception as e:
        report("RetinaFace.forward", "fail", str(e))
        traceback.print_exc()

        # Diagnose the specific issue
        if "reshape" in str(e).lower() or "shape" in str(e).lower():
            print("\n  >>> DIAGNOSIS: CixSession.run() returns flat 1D arrays.")
            print("  >>> RetinaFace expects shaped arrays matching get_outputs() shapes.")
            print("  >>> FIX: CixSession.run() must reshape outputs using INPUT_OUTPUT_MAPPING.\n")
    finally:
        sess.release()


# ──────────────────────────────────────────────────────────────
# 3. Test CLIP Visual
# ──────────────────────────────────────────────────────────────

def test_clip_visual():
    """Test CLIP visual encoder through the official pipeline."""
    from immich_ml.sessions.cix import CixSession

    model_path = CACHE / "clip/ViT-B-32__openai/visual/cix/model.cix"
    if not model_path.exists():
        report("CLIP.visual", "skip", "Model not found")
        return

    sess = CixSession(model_path)
    try:
        # Simulate what OpenClipVisualEncoder._predict does
        # Input: NCHW float32 image
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs = sess.run(None, {"image": dummy_input})

        print(f"  Visual output: {len(outputs)} tensors")
        for i, o in enumerate(outputs):
            print(f"    [{i}]: shape={o.shape}, dtype={o.dtype}, range=[{o.min():.4f}, {o.max():.4f}]")

        # OpenClipVisualEncoder does: res = self.session.run(None, self.transform(image))[0][0]
        # So outputs[0] should be (1, 512) and [0][0] gives (512,)
        assert outputs[0].shape == (512,) or (len(outputs[0].shape) == 1 and outputs[0].shape[0] == 512) or \
               (len(outputs[0].shape) == 2 and outputs[0].shape == (1, 512)), \
               f"Expected (1, 512) or (512,), got {outputs[0].shape}"

        # Check that [0][0] indexing works (pipeline uses this)
        embedding = outputs[0][0] if len(outputs[0].shape) == 2 else outputs[0]
        print(f"  Embedding: shape={embedding.shape}, norm={np.linalg.norm(embedding):.4f}")

        report("CLIP.visual", "pass" if len(outputs[0].shape) == 2 else "fail",
               f"output shape={outputs[0].shape} — pipeline expects [0][0] indexing on (1, 512)")
    except Exception as e:
        report("CLIP.visual", "fail", str(e))
        traceback.print_exc()
    finally:
        sess.release()


# ──────────────────────────────────────────────────────────────
# 4. Test CLIP Textual
# ──────────────────────────────────────────────────────────────

def test_clip_textual():
    """Test CLIP textual encoder through the official pipeline."""
    from immich_ml.sessions.cix import CixSession

    model_path = CACHE / "clip/ViT-B-32__openai/textual/cix/model.cix"
    if not model_path.exists():
        report("CLIP.textual", "skip", "Model not found")
        return

    sess = CixSession(model_path)
    try:
        # Simulate OpenClipTextualEncoder: tokens as int32
        dummy_tokens = np.array([[49406] + [0] * 75 + [49407]], dtype=np.int32)  # [CLS] + pad + [SEP]
        outputs = sess.run(None, {"text": dummy_tokens})

        print(f"  Textual output: {len(outputs)} tensors")
        for i, o in enumerate(outputs):
            print(f"    [{i}]: shape={o.shape}, dtype={o.dtype}, range=[{o.min():.4f}, {o.max():.4f}]")

        # Same indexing: outputs[0][0]
        assert outputs[0].shape == (512,) or outputs[0].shape == (1, 512), \
            f"Expected (1, 512), got {outputs[0].shape}"

        report("CLIP.textual", "pass" if len(outputs[0].shape) == 2 else "fail",
               f"output shape={outputs[0].shape}")
    except Exception as e:
        report("CLIP.textual", "fail", str(e))
        traceback.print_exc()
    finally:
        sess.release()


# ──────────────────────────────────────────────────────────────
# 5. Test Face Recognition (ArcFace wrapper)
# ──────────────────────────────────────────────────────────────

def test_face_recognition():
    """Test face recognition through _CixArcFaceWrapper."""
    import cv2
    from immich_ml.sessions.cix import CixSession

    model_path = CACHE / "facial-recognition/buffalo_l/recognition/cix/model.cix"
    if not model_path.exists():
        report("FaceRecognition", "skip", "Model not found")
        return

    sess = CixSession(model_path)
    try:
        # Inline _CixArcFaceWrapper to avoid rapidocr import chain
        class _CixArcFaceWrapper:
            def __init__(self, session):
                self.session = session
                self.input_mean = 127.5
                self.input_std = 127.5
                input_cfg = session.get_inputs()[0]
                self.input_size = tuple(input_cfg.shape[2:4][::-1])
                self.input_name = input_cfg.name
                self.output_names = [o.name for o in session.get_outputs()]

            def get_feat(self, imgs):
                if not isinstance(imgs, list):
                    imgs = [imgs]
                blob = cv2.dnn.blobFromImages(
                    imgs, 1.0 / self.input_std, self.input_size,
                    (self.input_mean, self.input_mean, self.input_mean), swapRB=True,
                )
                return self.session.run(self.output_names, {self.input_name: blob})[0]

        wrapper = _CixArcFaceWrapper(sess)

        # Create a dummy 112x112 face image
        dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        embedding = wrapper.get_feat([dummy_face])

        print(f"  ArcFace output: shape={embedding.shape}, norm={np.linalg.norm(embedding):.4f}")
        assert embedding.shape == (1, 512) or (len(embedding.shape) == 1 and embedding.shape[0] == 512), \
            f"Expected (1, 512), got {embedding.shape}"

        report("FaceRecognition", "pass" if embedding.shape == (1, 512) else "fail",
               f"output shape={embedding.shape}")
    except Exception as e:
        report("FaceRecognition", "fail", str(e))
        traceback.print_exc()
    finally:
        sess.release()


# ──────────────────────────────────────────────────────────────
# 6. Test output reshape issue (core blocker diagnosis)
# ──────────────────────────────────────────────────────────────

def test_detection_output_shapes():
    """Test whether CixSession.run() output shapes match what RetinaFace expects."""
    import cv2
    from immich_ml.sessions.cix import CixSession, INPUT_OUTPUT_MAPPING

    det_path = CACHE / "facial-recognition/buffalo_l/detection/cix/model.cix"
    if not det_path.exists():
        report("Detection.output_shapes", "skip", "Detection model not found")
        return

    sess = CixSession(det_path)
    try:
        # Create blob like RetinaFace.forward() does
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        blob = cv2.dnn.blobFromImage(img, 1.0 / 128.0, (640, 640), (127.5, 127.5, 127.5), swapRB=True)

        outputs = sess.run(None, {"input.1": blob})

        expected_shapes = list(INPUT_OUTPUT_MAPPING["detection"]["output"].values())
        expected_names = list(INPUT_OUTPUT_MAPPING["detection"]["output"].keys())

        print(f"  Got {len(outputs)} output tensors:")
        all_shapes_ok = True
        for i, out in enumerate(outputs):
            expected = expected_shapes[i] if i < len(expected_shapes) else "??"
            expected_total = np.prod(expected) if i < len(expected_shapes) else -1
            actual_total = out.size
            name = expected_names[i] if i < len(expected_names) else f"unknown_{i}"
            shape_match = out.shape == expected if isinstance(expected, tuple) else False

            status = "✓" if shape_match else f"✗ (flat {out.shape})"
            print(f"    [{i}] {name}: actual={out.shape} (size={actual_total}), "
                  f"expected={expected} (size={expected_total}) {status}")
            if not shape_match:
                all_shapes_ok = False

        if all_shapes_ok:
            report("Detection.output_shapes", "pass", "All outputs correctly shaped")
        else:
            # Check if total sizes match (just need reshape)
            sizes_match = all(
                outputs[i].size == np.prod(expected_shapes[i])
                for i in range(min(len(outputs), len(expected_shapes)))
            )
            if sizes_match:
                report("Detection.output_shapes", "fail",
                       "Outputs are FLAT 1D — need reshape in CixSession.run(). "
                       "Total sizes match, so adding reshape to run() will fix this.")
            else:
                report("Detection.output_shapes", "fail",
                       "Output sizes don't match expected! Possible tensor ordering mismatch.")

    except Exception as e:
        report("Detection.output_shapes", "fail", str(e))
        traceback.print_exc()
    finally:
        sess.release()


# ──────────────────────────────────────────────────────────────
# 7. Test OCR feasibility
# ──────────────────────────────────────────────────────────────

def test_ocr_feasibility():
    """Check if OCR can work with CIX models through official pipeline."""
    from immich_ml.sessions.cix import CixSession

    det_path = CACHE / "ocr/PP-OCRv4_mobile/detection/cix/model.cix"
    rec_path = CACHE / "ocr/PP-OCRv4_mobile/recognition/cix/model.cix"

    if not det_path.exists():
        report("OCR.detection_model", "skip", "OCR detection CIX model not found")
    else:
        try:
            sess = CixSession(det_path)
            inputs = sess.get_inputs()
            outputs = sess.get_outputs()
            print(f"  OCR Det: input={inputs[0].name} {inputs[0].shape}, output={outputs[0].name} {outputs[0].shape}")
            sess.release()
            report("OCR.detection_model", "pass", "Model loads, but TextDetector hardcodes OrtSession (needs code change)")
        except Exception as e:
            report("OCR.detection_model", "fail", str(e))

    if not rec_path.exists():
        report("OCR.recognition_model", "skip", "OCR recognition CIX model not found")
    else:
        try:
            sess = CixSession(rec_path)
            inputs = sess.get_inputs()
            outputs = sess.get_outputs()
            print(f"  OCR Rec: input={inputs[0].name} {inputs[0].shape}, output={outputs[0].name} {outputs[0].shape}")
            sess.release()
            report("OCR.recognition_model", "pass",
                   "Model loads, but TextRecognizer uses RapidOCR internals that need ORT session directly")
        except Exception as e:
            report("OCR.recognition_model", "fail", str(e))

    # Code analysis
    print("\n  OCR Pipeline Analysis:")
    print("  - TextDetector._load() hardcodes `OrtSession(self.model_path)` — needs `self._make_session()`")
    print("  - TextDetector.__init__ forces `model_format=ModelFormat.ONNX` — needs CIX override")
    print("  - TextRecognizer uses RapidOCR's TextRecognizer which wraps ORT session directly")
    print("  - TextRecognizer passes `session=session.session` (ORT InferenceSession) to RapidOCR")
    print("  - PP-OCRv4 (CIX) vs PP-OCRv5 (upstream) — different model/dict versions")
    print("  - OCR input is dynamic-size (resized to multiples of 32) — CIX model has fixed 960x608")
    report("OCR.pipeline_feasibility", "fail",
           "Multiple blockers: hardcoded ONNX format, RapidOCR needs ORT session, "
           "dynamic vs fixed input size, PP-OCRv4 vs v5 mismatch")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Immich ML — CIX NPU Official Pipeline Compatibility Test")
    print("=" * 70)
    print()

    # 1. Basic CixSession check
    print("── 1. CixSession Basic ──")
    if not test_cix_session_basic():
        print("\nCIX NPU not available, cannot continue.\n")
        return 1
    print()

    # 2. Detection output shapes (core blocker diagnosis)
    print("── 2. Detection Output Shapes ──")
    test_detection_output_shapes()
    print()

    # 3. RetinaFace compatibility
    print("── 3. RetinaFace Compatibility ──")
    test_retinaface_inference()
    print()

    # 4. CLIP Visual
    print("── 4. CLIP Visual ──")
    test_clip_visual()
    print()

    # 5. CLIP Textual
    print("── 5. CLIP Textual ──")
    test_clip_textual()
    print()

    # 6. Face Recognition
    print("── 6. Face Recognition ──")
    test_face_recognition()
    print()

    # 7. OCR
    print("── 7. OCR Feasibility ──")
    test_ocr_feasibility()
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passes = sum(1 for v in RESULTS.values() if v == "pass")
    fails = sum(1 for v in RESULTS.values() if v == "fail")
    skips = sum(1 for v in RESULTS.values() if v == "skip")
    print(f"  Pass: {passes}  |  Fail: {fails}  |  Skip: {skips}")
    print()
    for name, status in RESULTS.items():
        tag = "✅" if status == "pass" else "❌" if status == "fail" else "⚠️ "
        print(f"  {tag} {name}")
    print()

    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
