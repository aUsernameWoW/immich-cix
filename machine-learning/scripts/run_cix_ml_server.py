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

# Supported CIX NPU models
CIX_SUPPORTED_OCR_MODELS = {"PP-OCRv4-cix", "PP-OCRv4_mobile"}

# Global sessions
clip_visual_session = None
clip_text_session = None
clip_tokenizer = None
face_det_session = None
face_rec_session = None
ocr_det_session = None
ocr_rec_session = None
ocr_char_dict = None


def load_models():
    """Load all CIX NPU models."""
    global clip_visual_session, clip_text_session, clip_tokenizer
    global face_det_session, face_rec_session
    global ocr_det_session, ocr_rec_session, ocr_char_dict

    cache = Path.home() / ".cache/immich_ml"

    print("Loading CIX NPU models...")

    # CLIP Visual
    path = cache / "clip/ViT-B-32__openai/visual/cix/model.cix"
    if path.exists():
        try:
            clip_visual_session = CixSession(path)
            print(f"  Loaded CLIP Visual: {path}")
        except Exception as e:
            print(f"  Failed to load CLIP Visual: {e}")

    # CLIP Text
    path = cache / "clip/ViT-B-32__openai/textual/cix/model.cix"
    if path.exists():
        try:
            clip_text_session = CixSession(path)
            print(f"  Loaded CLIP Text: {path}")
        except Exception as e:
            print(f"  Failed to load CLIP Text: {e}")

    # CLIP Tokenizer
    tokenizer_path = cache / "clip/ViT-B-32__openai/textual/tokenizer.json"
    tokenizer_cfg_path = cache / "clip/ViT-B-32__openai/textual/tokenizer_config.json"
    if tokenizer_path.exists():
        clip_tokenizer = Tokenizer.from_file(str(tokenizer_path))
        # Load config for context_length and pad_token
        if tokenizer_cfg_path.exists():
            with open(tokenizer_cfg_path, encoding='utf-8') as f:
                cfg = json.load(f)
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
        try:
            face_det_session = CixSession(path)
            print(f"  Loaded Face Detection: {path}")
        except Exception as e:
            print(f"  Failed to load Face Detection: {e}")

    # Face Recognition
    path = cache / "facial-recognition/buffalo_l/recognition/cix/model.cix"
    if path.exists():
        try:
            face_rec_session = CixSession(path)
            print(f"  Loaded Face Recognition: {path}")
        except Exception as e:
            print(f"  Failed to load Face Recognition: {e}")

    # OCR Detection (PP-OCRv4)
    path = cache / "ocr/PP-OCRv4_mobile/detection/cix/model.cix"
    if path.exists():
        try:
            ocr_det_session = CixSession(path)
            print(f"  Loaded OCR Detection: {path}")
        except Exception as e:
            print(f"  Failed to load OCR Detection: {e}")

    # OCR Recognition (PP-OCRv4)
    path = cache / "ocr/PP-OCRv4_mobile/recognition/cix/model.cix"
    if path.exists():
        try:
            ocr_rec_session = CixSession(path)
            print(f"  Loaded OCR Recognition: {path}")
        except Exception as e:
            print(f"  Failed to load OCR Recognition: {e}")

    # OCR Character Dictionary
    dict_path = cache / "ocr/PP-OCRv4_mobile/ppocr_keys_v1.txt"
    if dict_path.exists():
        with open(dict_path, 'r', encoding='utf-8') as f:
            # Add blank token at index 0 for CTC decoding
            ocr_char_dict = [''] + [line.strip() for line in f.readlines()]
            # Add space at end
            ocr_char_dict.append(' ')
        print(f"  Loaded OCR Dictionary: {len(ocr_char_dict)} characters")

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
    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

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
    """Preprocess image for SCRFD face detection using cv2.dnn.blobFromImage."""
    # Convert PIL to cv2 BGR format
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    orig_h, orig_w = img_cv2.shape[:2]

    input_size = (640, 640)
    im_ratio = orig_h / orig_w
    model_ratio = input_size[1] / input_size[0]

    if im_ratio > model_ratio:
        new_height = input_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = input_size[0]
        new_height = int(new_width * im_ratio)

    det_scale = new_height / orig_h
    resized = cv2.resize(img_cv2, (new_width, new_height))

    # Pad to 640x640
    det_image = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    det_image[:new_height, :new_width] = resized

    # Create blob: (x - mean) / std, swapRB
    blob = cv2.dnn.blobFromImage(det_image, 1.0/128.0, input_size, (127.5, 127.5, 127.5), swapRB=True)

    return blob, (orig_w, orig_h), det_scale


# SCRFD anchor cache
_scrfd_anchor_cache: dict = {}


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Decode distance predictions to bounding boxes."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Decode distance predictions to keypoints."""
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def _nms(dets: np.ndarray, iou_thres: float = 0.4) -> list[int]:
    """Non-Maximum Suppression."""
    if len(dets) == 0:
        return []
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        ovr = w * h / (areas[i] + areas[order[1:]] - w * h)
        order = order[np.where(ovr <= iou_thres)[0] + 1]
    return keep


def decode_scrfd_output(outputs: list, orig_size: tuple[int, int], det_scale: float,
                        score_thresh: float = 0.5, iou_thresh: float = 0.4) -> dict:
    """
    Decode SCRFD face detection outputs.

    Args:
        outputs: List of 9 output tensors from CIX NPU
        orig_size: Original image size (width, height)
        det_scale: Scale factor used during preprocessing
        score_thresh: Confidence threshold
        iou_thresh: NMS IoU threshold

    Returns:
        dict with 'boxes', 'scores', 'landmarks' arrays
    """
    global _scrfd_anchor_cache

    input_size = 640
    strides = [8, 16, 32]
    num_anchors = 2

    # Reshape outputs: [scores_8, scores_16, scores_32, bboxes_8, bboxes_16, bboxes_32, kps_8, kps_16, kps_32]
    output_reshape = [1, 1, 1, 4, 4, 4, 10, 10, 10]
    reshaped = []
    for j in range(9):
        reshaped.append(np.reshape(outputs[j], (-1, output_reshape[j])))

    scores_list, bboxes_list, kpss_list = [], [], []

    for idx, stride in enumerate(strides):
        scores = reshaped[idx]
        bbox_preds = reshaped[idx + 3] * stride
        kps_preds = reshaped[idx + 6] * stride

        feat_h = input_size // stride
        feat_w = input_size // stride
        key = (feat_h, feat_w, stride)

        if key not in _scrfd_anchor_cache:
            anchor_centers = np.stack(np.mgrid[:feat_h, :feat_w][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))
            _scrfd_anchor_cache[key] = anchor_centers
        anchor_centers = _scrfd_anchor_cache[key]

        pos_inds = np.where(scores >= score_thresh)[0]
        if len(pos_inds) > 0:
            bboxes = _distance2bbox(anchor_centers, bbox_preds)
            kpss = _distance2kps(anchor_centers, kps_preds).reshape((-1, 5, 2))
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            kpss_list.append(kpss[pos_inds])

    if not scores_list:
        return {
            'boxes': np.array([]).reshape(0, 4),
            'scores': np.array([]),
            'landmarks': np.array([]).reshape(0, 5, 2),
        }

    scores = np.vstack(scores_list)
    bboxes = np.vstack(bboxes_list) / det_scale
    kpss = np.vstack(kpss_list) / det_scale

    # NMS
    pre_det = np.hstack((bboxes, scores)).astype(np.float32)
    order = scores.ravel().argsort()[::-1]
    pre_det, kpss = pre_det[order], kpss[order]

    keep = _nms(pre_det, iou_thresh)

    return {
        'boxes': pre_det[keep, :4],
        'scores': pre_det[keep, 4],
        'landmarks': kpss[keep],
    }


def serialize_embedding(arr: np.ndarray) -> str:
    """Serialize numpy array to JSON string (matching Immich format)."""
    return orjson.dumps(arr.flatten().tolist()).decode()


# ============== OCR Functions ==============

def preprocess_ocr_detection(image: Image.Image) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    """
    Preprocess image for PP-OCRv4 text detection.

    Args:
        image: PIL Image (RGB)

    Returns:
        Tuple of (preprocessed tensor, original size, shape_list for postprocess)
    """
    # Target size for CIX model: 960x608
    target_h, target_w = 960, 608

    orig_w, orig_h = image.size

    # Resize keeping aspect ratio, pad to target size
    ratio = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)

    # Ensure dimensions are multiples of 32
    new_w = max(int(round(new_w / 32) * 32), 32)
    new_h = max(int(round(new_h / 32) * 32), 32)

    # Resize
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    img_np = np.array(resized, dtype=np.float32)

    # Convert RGB to BGR
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Normalize: (x - mean) / std, with mean=[0.485, 0.456, 0.406]*255, std=[0.229, 0.224, 0.225]*255
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
    img_np = (img_np - mean) / std

    # Pad to target size
    padded = np.zeros((target_h, target_w, 3), dtype=np.float32)
    padded[:new_h, :new_w, :] = img_np

    # NCHW format
    padded = padded.transpose(2, 0, 1)
    padded = np.expand_dims(padded, 0)

    # Shape list for postprocessing: [src_h, src_w, ratio_h, ratio_w]
    shape_list = np.array([[orig_h, orig_w, new_h / orig_h, new_w / orig_w]], dtype=np.float32)

    return padded.astype(np.float32), (orig_w, orig_h), shape_list


def postprocess_ocr_detection(
    pred: np.ndarray,
    shape_list: np.ndarray,
    thresh: float = 0.3,
    box_thresh: float = 0.5,
    unclip_ratio: float = 1.6,
    max_candidates: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Postprocess PP-OCR text detection output (DBPostProcess).

    Args:
        pred: Model output [1, 1, H, W]
        shape_list: Original shape info [1, 4]
        thresh: Binarization threshold
        box_thresh: Minimum box score threshold
        unclip_ratio: Ratio to expand text boxes
        max_candidates: Maximum number of text boxes

    Returns:
        Tuple of (boxes [N, 4, 2], scores [N])
    """
    import pyclipper
    from shapely.geometry import Polygon

    pred = pred[:, 0, :, :]  # [1, H, W]
    segmentation = pred > thresh

    src_h, src_w = int(shape_list[0, 0]), int(shape_list[0, 1])
    bitmap = segmentation[0]
    height, width = bitmap.shape

    # Find contours
    contours, _ = cv2.findContours(
        (bitmap * 255).astype(np.uint8),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes, scores = [], []
    for contour in contours[:max_candidates]:
        # Get minimum area rectangle
        points, sside = _get_mini_boxes(contour)
        if sside < 3:
            continue

        points = np.array(points)

        # Calculate box score
        score = _box_score_fast(pred[0], points)
        if score < box_thresh:
            continue

        # Unclip (expand) the box
        try:
            poly = Polygon(points)
            distance = poly.area * unclip_ratio / poly.length
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(points.astype(int).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = offset.Execute(distance)
            if not expanded:
                continue
            box = np.array(expanded[0])
        except Exception:
            continue

        # Get final mini box
        box, sside = _get_mini_boxes(box.reshape((-1, 1, 2)))
        if sside < 3 + 2:
            continue

        box = np.array(box)
        # Scale back to original image coordinates
        box[:, 0] = np.clip(np.round(box[:, 0] / width * src_w), 0, src_w)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * src_h), 0, src_h)

        boxes.append(box.astype(np.float32))
        scores.append(score)

    if not boxes:
        return np.array([]).reshape(0, 4, 2), np.array([])

    return np.array(boxes), np.array(scores)


def _get_mini_boxes(contour: np.ndarray) -> tuple[list, float]:
    """Get minimum area bounding box from contour."""
    bounding_box = cv2.minAreaRect(contour.astype(np.int32))
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    if points[1][1] > points[0][1]:
        index_1, index_4 = 0, 1
    else:
        index_1, index_4 = 1, 0

    if points[3][1] > points[2][1]:
        index_2, index_3 = 2, 3
    else:
        index_2, index_3 = 3, 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def _box_score_fast(bitmap: np.ndarray, box: np.ndarray) -> float:
    """Calculate score for a text box."""
    h, w = bitmap.shape[:2]
    box = box.copy()
    xmin = np.clip(int(np.floor(box[:, 0].min())), 0, w - 1)
    xmax = np.clip(int(np.ceil(box[:, 0].max())), 0, w - 1)
    ymin = np.clip(int(np.floor(box[:, 1].min())), 0, h - 1)
    ymax = np.clip(int(np.ceil(box[:, 1].max())), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def sorted_ocr_boxes(boxes: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Sort text boxes from top to bottom, left to right."""
    if len(boxes) == 0:
        return boxes, []

    # Sort by y, then by x within same line
    sorted_boxes = sorted(enumerate(boxes), key=lambda x: (x[1][0][1], x[1][0][0]))
    result = list(sorted_boxes)

    # Fine-tune: swap adjacent boxes if they're on same line but wrong x order
    for i in range(len(result) - 1):
        for j in range(i, -1, -1):
            if abs(result[j + 1][1][0][1] - result[j][1][0][1]) < 10 and \
                    result[j + 1][1][0][0] < result[j][1][0][0]:
                result[j], result[j + 1] = result[j + 1], result[j]
            else:
                break

    indices = [r[0] for r in result]
    return np.array([r[1] for r in result]), indices


def get_rotate_crop_image(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Crop and rotate text region from image."""
    points = points.astype(np.float32)

    img_crop_width = int(max(
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[2] - points[3])
    ))
    img_crop_height = int(max(
        np.linalg.norm(points[0] - points[3]),
        np.linalg.norm(points[1] - points[2])
    ))

    pts_std = np.float32([
        [0, 0],
        [img_crop_width, 0],
        [img_crop_width, img_crop_height],
        [0, img_crop_height]
    ])

    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img, M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC
    )

    # Rotate if height > 1.5 * width
    if dst_img.shape[0] * 1.0 / dst_img.shape[1] >= 1.5:
        dst_img = np.rot90(dst_img)

    return dst_img


def preprocess_ocr_recognition(img: np.ndarray) -> np.ndarray:
    """
    Preprocess cropped text image for PP-OCRv4 recognition.

    Args:
        img: Cropped text image (BGR, uint8)

    Returns:
        Preprocessed tensor [1, 3, 32, 400]
    """
    # Target shape: [3, 32, 400]
    imgC, imgH, imgW = 3, 32, 400

    h, w = img.shape[:2]
    ratio = w / float(h)

    if np.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(np.ceil(imgH * ratio))

    resized_img = cv2.resize(img, (resized_w, imgH))
    resized_img = resized_img.astype(np.float32)

    # Normalize
    resized_img = resized_img.transpose((2, 0, 1)) / 255.0
    resized_img -= 0.5
    resized_img /= 0.5

    # Pad to target width
    padding = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding[:, :, :resized_w] = resized_img

    return np.expand_dims(padding, 0)


def postprocess_ocr_recognition(pred: np.ndarray, char_dict: list[str]) -> tuple[str, float]:
    """
    CTC decode recognition output.

    Args:
        pred: Model output [1, T, num_classes] (after softmax)
        char_dict: Character dictionary

    Returns:
        Tuple of (decoded text, confidence score)
    """
    # Get best path
    pred_idx = pred.argmax(axis=2)[0]  # [T]
    pred_prob = pred.max(axis=2)[0]    # [T]

    # CTC decode: remove blanks and duplicates
    char_list = []
    conf_list = []
    prev_idx = 0  # blank

    for i, idx in enumerate(pred_idx):
        if idx != 0 and idx != prev_idx:  # Not blank and not duplicate
            if idx < len(char_dict):
                char_list.append(char_dict[idx])
                conf_list.append(pred_prob[i])
        prev_idx = idx

    text = ''.join(char_list)
    confidence = float(np.mean(conf_list)) if conf_list else 0.0

    return text, confidence


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax along specified axis."""
    x_max = np.max(x, axis=axis, keepdims=True)
    x_exp = np.exp(x - x_max)
    return x_exp / np.sum(x_exp, axis=axis, keepdims=True)


# Reference landmarks for face alignment (ArcFace standard)
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def crop_aligned_face(image: np.ndarray, landmarks: np.ndarray, image_size: int = 112) -> np.ndarray | None:
    """
    Crop and align face using 5-point landmarks.

    Args:
        image: Input image (RGB)
        landmarks: 5x2 array of facial landmarks
        image_size: Output size (112 for ArcFace)

    Returns:
        Aligned face image or None if alignment fails
    """
    try:
        # Scale reference landmarks to target size
        dst = ARCFACE_DST * (image_size / 112.0)
        src = landmarks.astype(np.float32)

        # Estimate similarity transform
        from skimage.transform import SimilarityTransform
        tform = SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        # Warp face
        warped = cv2.warpAffine(image, M, (image_size, image_size), borderValue=0)
        return warped
    except Exception as e:
        print(f"Face alignment error: {e}")
        return None


def preprocess_face_recognition(face: np.ndarray) -> np.ndarray:
    """
    Preprocess aligned face for ArcFace model.

    Args:
        face: Aligned face image (112x112 RGB)

    Returns:
        Preprocessed tensor (1, 3, 112, 112)
    """
    # Convert RGB to BGR for ArcFace
    face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

    # Normalize: (x - 127.5) / 127.5
    face_np = face_bgr.astype(np.float32)
    face_np = (face_np - 127.5) / 127.5

    # NCHW format
    face_np = face_np.transpose(2, 0, 1)
    return np.expand_dims(face_np, 0).astype(np.float32)


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
            "ocr_detection": ocr_det_session is not None,
            "ocr_recognition": ocr_rec_session is not None,
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

            # Face Detection + Recognition (combined)
            elif task == "facial-recognition" and type_name == "detection":
                if face_det_session and pil_image:
                    start = time.perf_counter()

                    # Run face detection
                    blob, orig_size, det_scale = preprocess_face_detection(pil_image)
                    det_outputs = face_det_session.run(None, {"input.1": blob})

                    min_score = options.get("minScore", 0.7)
                    detection = decode_scrfd_output(det_outputs, orig_size, det_scale, min_score)

                    det_elapsed = (time.perf_counter() - start) * 1000

                    # Process each detected face
                    faces_result = []
                    img_np = np.array(pil_image)

                    for i, (box, score, landmarks) in enumerate(zip(
                        detection['boxes'], detection['scores'], detection['landmarks']
                    )):
                        x1, y1, x2, y2 = box.astype(int)

                        # Get face embedding if recognition model is available
                        embedding_str = "[]"
                        if face_rec_session:
                            try:
                                # Align and crop face using landmarks
                                face_crop = crop_aligned_face(img_np, landmarks)
                                if face_crop is not None:
                                    # Preprocess for ArcFace
                                    face_tensor = preprocess_face_recognition(face_crop)
                                    rec_outputs = face_rec_session.run(None, {"input.1": face_tensor})
                                    embedding = rec_outputs[0].flatten()
                                    # Normalize embedding
                                    embedding = embedding / np.linalg.norm(embedding)
                                    embedding_str = orjson.dumps(embedding.tolist()).decode()
                            except Exception as e:
                                print(f"Face recognition error for face {i}: {e}")

                        faces_result.append({
                            "boundingBox": {
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2),
                            },
                            "embedding": embedding_str,
                            "score": float(score),
                        })

                    elapsed = (time.perf_counter() - start) * 1000
                    print(f"Face Detection+Recognition: {elapsed:.1f}ms (det: {det_elapsed:.1f}ms), found {len(faces_result)} faces")

                    response["facial-recognition"] = faces_result
                else:
                    response["facial-recognition"] = []

            # Face Recognition (standalone - for face crops)
            elif task == "facial-recognition" and type_name == "recognition":
                # Skip if detection already set the response (we do recognition in detection handler)
                # This is only used for standalone face embedding requests with pre-cropped faces
                if "facial-recognition" not in response:
                    response["facial-recognition"] = []

            # OCR Detection + Recognition
            elif task == "ocr" and type_name == "detection":
                if ocr_det_session and ocr_rec_session and pil_image and ocr_char_dict:
                    # Validate model name - CIX NPU only supports PP-OCRv4
                    if model_name and model_name not in CIX_SUPPORTED_OCR_MODELS:
                        print(f"OCR: Requested model '{model_name}' not supported on CIX NPU, using PP-OCRv4-cix")

                    start = time.perf_counter()

                    # Get options
                    min_det_score = options.get("minScore", 0.5)

                    # 1. Text Detection
                    det_input, orig_size, shape_list = preprocess_ocr_detection(pil_image)
                    det_output = ocr_det_session.run(None, {"x": det_input})

                    # Reshape output: should be [1, 1, 960, 608]
                    det_pred = det_output[0].reshape(1, 1, 960, 608)

                    # Postprocess detection
                    boxes, box_scores = postprocess_ocr_detection(
                        det_pred, shape_list, box_thresh=min_det_score
                    )

                    det_elapsed = (time.perf_counter() - start) * 1000

                    if len(boxes) == 0:
                        response["ocr"] = {
                            "text": [],
                            "box": [],
                            "boxScore": [],
                            "textScore": [],
                        }
                        print(f"OCR: {det_elapsed:.1f}ms, no text detected")
                    else:
                        # Sort boxes in reading order
                        sorted_boxes_arr, _ = sorted_ocr_boxes(boxes)

                        # 2. Text Recognition for each box
                        img_np = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                        texts = []
                        text_scores = []
                        valid_boxes = []
                        valid_box_scores = []

                        rec_start = time.perf_counter()
                        for i, box in enumerate(sorted_boxes_arr):
                            try:
                                # Crop text region
                                crop_img = get_rotate_crop_image(img_np, box)
                                if crop_img.shape[0] < 2 or crop_img.shape[1] < 2:
                                    continue

                                # Preprocess for recognition
                                rec_input = preprocess_ocr_recognition(crop_img)

                                # Run recognition
                                rec_output = ocr_rec_session.run(None, {"x": rec_input})

                                # Reshape and apply softmax: [1, 100, 6625]
                                rec_pred = rec_output[0].reshape(1, 100, 6625)
                                rec_pred = softmax(rec_pred, axis=2)

                                # Decode text
                                text, conf = postprocess_ocr_recognition(rec_pred, ocr_char_dict)

                                if text and conf > 0.1:  # Filter very low confidence
                                    texts.append(text)
                                    text_scores.append(conf)
                                    # Normalize box coordinates to [0, 1]
                                    norm_box = box.copy()
                                    norm_box[:, 0] /= orig_size[0]  # width
                                    norm_box[:, 1] /= orig_size[1]  # height
                                    valid_boxes.append(norm_box)
                                    valid_box_scores.append(box_scores[i] if i < len(box_scores) else 0.5)

                            except Exception as e:
                                print(f"OCR recognition error for box {i}: {e}")
                                continue

                        rec_elapsed = (time.perf_counter() - rec_start) * 1000
                        total_elapsed = (time.perf_counter() - start) * 1000

                        # Format response matching Immich API
                        response["ocr"] = {
                            "text": texts,
                            "box": np.array(valid_boxes).reshape(-1).tolist() if valid_boxes else [],
                            "boxScore": valid_box_scores,
                            "textScore": text_scores,
                        }

                        print(f"OCR: {total_elapsed:.1f}ms (det: {det_elapsed:.1f}ms, rec: {rec_elapsed:.1f}ms), "
                              f"found {len(texts)} text regions")
                else:
                    if "ocr" not in response:
                        response["ocr"] = {
                            "text": [],
                            "box": [],
                            "boxScore": [],
                            "textScore": [],
                        }

            # OCR Recognition (standalone - handled in detection)
            elif task == "ocr" and type_name == "recognition":
                if "ocr" not in response:
                    response["ocr"] = {
                        "text": [],
                        "box": [],
                        "boxScore": [],
                        "textScore": [],
                    }

    return ORJSONResponse(response)


if __name__ == "__main__":
    print(f"CIX NPU available: {is_available}")
    load_models()
    uvicorn.run(app, host="0.0.0.0", port=3003)
