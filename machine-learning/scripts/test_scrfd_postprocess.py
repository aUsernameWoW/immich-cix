#!/usr/bin/env python3
"""Test SCRFD post-processing implementation."""

import numpy as np
from pathlib import Path
from PIL import Image
import cv2

# SCRFD configuration for 500M model
# Input size: 640x640
# Strides: [8, 16, 32]
# Anchors per location: 2

def sigmoid(x):
    """Sigmoid activation."""
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def generate_anchor_centers(input_size: int = 640, strides: list = [8, 16, 32], num_anchors: int = 2):
    """
    Generate anchor centers for all feature map levels.
    
    Returns:
        List of anchor centers for each stride, shape (H*W*num_anchors, 2)
    """
    anchor_centers_list = []
    
    for stride in strides:
        feat_h = input_size // stride
        feat_w = input_size // stride
        
        # Create grid of (x, y) coordinates
        # mgrid returns [y_coords, x_coords], we want x first
        y_grid, x_grid = np.mgrid[:feat_h, :feat_w]
        
        # Stack to (H, W, 2) with (x, y) order, then multiply by stride
        # Add 0.5 to center the anchors in each cell, then scale
        anchor_centers = np.stack([x_grid, y_grid], axis=-1).astype(np.float32)
        anchor_centers = (anchor_centers + 0.5) * stride
        
        # Reshape to (H*W, 2)
        anchor_centers = anchor_centers.reshape(-1, 2)
        
        # Repeat for num_anchors (each location has multiple anchors)
        anchor_centers = np.tile(anchor_centers, (num_anchors, 1))
        # Actually SCRFD interleaves anchors: [a1_loc1, a2_loc1, a1_loc2, a2_loc2, ...]
        # So we need to reshape properly
        anchor_centers = anchor_centers.reshape(num_anchors, -1, 2).transpose(1, 0, 2).reshape(-1, 2)
        
        anchor_centers_list.append(anchor_centers)
    
    return anchor_centers_list


def distance2bbox(anchor_centers: np.ndarray, distances: np.ndarray, max_shape=None):
    """
    Decode distance predictions to bounding boxes.
    
    Args:
        anchor_centers: (N, 2) array of (x, y) anchor centers
        distances: (N, 4) array of (left, top, right, bottom) distances
        max_shape: Optional (height, width) to clip boxes
    
    Returns:
        (N, 4) array of (x1, y1, x2, y2) boxes
    """
    x1 = anchor_centers[:, 0] - distances[:, 0]
    y1 = anchor_centers[:, 1] - distances[:, 1]
    x2 = anchor_centers[:, 0] + distances[:, 2]
    y2 = anchor_centers[:, 1] + distances[:, 3]
    
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(anchor_centers: np.ndarray, distances: np.ndarray, max_shape=None):
    """
    Decode distance predictions to keypoints.
    
    Args:
        anchor_centers: (N, 2) array of (x, y) anchor centers
        distances: (N, 10) array of keypoint offsets
        max_shape: Optional (height, width) to clip keypoints
    
    Returns:
        (N, 5, 2) array of keypoint coordinates
    """
    num_kps = distances.shape[1] // 2
    kps = []
    
    for i in range(num_kps):
        px = anchor_centers[:, 0] + distances[:, i * 2]
        py = anchor_centers[:, 1] + distances[:, i * 2 + 1]
        
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        
        kps.append(np.stack([px, py], axis=-1))
    
    return np.stack(kps, axis=1)  # (N, 5, 2)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.4):
    """
    Non-Maximum Suppression.
    
    Args:
        boxes: (N, 4) array of (x1, y1, x2, y2) boxes
        scores: (N,) array of confidence scores
        iou_threshold: IoU threshold for suppression
    
    Returns:
        Indices of kept boxes
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Compute IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Keep boxes with IoU below threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep, dtype=np.int32)


def decode_scrfd_outputs(
    outputs: list[np.ndarray],
    input_size: int = 640,
    orig_size: tuple[int, int] = None,
    scale: float = 1.0,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.4,
) -> dict:
    """
    Decode SCRFD model outputs to face detections.
    
    Args:
        outputs: List of 9 output tensors from CIX NPU
        input_size: Model input size (640)
        orig_size: Original image size (height, width)
        scale: Scale factor used during preprocessing
        score_threshold: Minimum confidence threshold
        iou_threshold: NMS IoU threshold
    
    Returns:
        Dictionary with 'boxes', 'scores', 'landmarks' keys
    """
    strides = [8, 16, 32]
    num_anchors = 2
    
    # Reshape flattened outputs
    # Output order: scores_8, scores_16, scores_32, bboxes_8, bboxes_16, bboxes_32, kps_8, kps_16, kps_32
    scores_list = []
    bboxes_list = []
    kps_list = []
    
    for i, stride in enumerate(strides):
        feat_size = input_size // stride
        num_points = feat_size * feat_size * num_anchors
        
        # Scores: apply sigmoid
        scores = sigmoid(outputs[i].reshape(-1))
        scores_list.append(scores)
        
        # Bboxes: reshape to (N, 4)
        bboxes = outputs[3 + i].reshape(-1, 4)
        bboxes_list.append(bboxes)
        
        # Keypoints: reshape to (N, 10)
        kps = outputs[6 + i].reshape(-1, 10)
        kps_list.append(kps)
    
    # Generate anchor centers
    anchor_centers_list = generate_anchor_centers(input_size, strides, num_anchors)
    
    # Process each stride level
    all_boxes = []
    all_scores = []
    all_kps = []
    
    for i, stride in enumerate(strides):
        scores = scores_list[i]
        bboxes = bboxes_list[i]
        kps = kps_list[i]
        anchor_centers = anchor_centers_list[i]
        
        # Filter by score threshold
        mask = scores > score_threshold
        if not mask.any():
            continue
        
        scores = scores[mask]
        bboxes = bboxes[mask]
        kps = kps[mask]
        anchor_centers = anchor_centers[mask]
        
        # Decode boxes and keypoints
        # SCRFD outputs are already scaled by stride internally
        decoded_boxes = distance2bbox(anchor_centers, bboxes * stride, max_shape=(input_size, input_size))
        decoded_kps = distance2kps(anchor_centers, kps * stride, max_shape=(input_size, input_size))
        
        all_boxes.append(decoded_boxes)
        all_scores.append(scores)
        all_kps.append(decoded_kps)
    
    if len(all_boxes) == 0:
        return {
            'boxes': np.array([]).reshape(0, 4),
            'scores': np.array([]),
            'landmarks': np.array([]).reshape(0, 5, 2),
        }
    
    # Concatenate all levels
    all_boxes = np.concatenate(all_boxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_kps = np.concatenate(all_kps, axis=0)
    
    # Apply NMS
    keep_indices = nms(all_boxes, all_scores, iou_threshold)
    
    boxes = all_boxes[keep_indices]
    scores = all_scores[keep_indices]
    landmarks = all_kps[keep_indices]
    
    # Scale back to original image coordinates
    if orig_size is not None and scale != 1.0:
        boxes = boxes / scale
        landmarks = landmarks / scale
    
    return {
        'boxes': boxes,
        'scores': scores,
        'landmarks': landmarks,
    }


def preprocess_image(image: Image.Image, input_size: int = 640):
    """
    Preprocess image for SCRFD.
    
    Returns:
        input_tensor: (1, 3, H, W) normalized tensor
        orig_size: (height, width) of original image
        scale: scale factor used
    """
    orig_w, orig_h = image.size
    
    # Calculate scale to fit in input_size
    scale = min(input_size / orig_w, input_size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    
    # Resize
    image = image.resize((new_w, new_h), Image.BILINEAR)
    
    # Pad to input_size x input_size
    padded = Image.new("RGB", (input_size, input_size), (0, 0, 0))
    padded.paste(image, (0, 0))
    
    # Normalize: (x - 127.5) / 128.0
    img_np = np.array(padded, dtype=np.float32)
    img_np = (img_np - 127.5) / 128.0
    
    # To NCHW
    img_np = img_np.transpose(2, 0, 1)
    input_tensor = np.expand_dims(img_np, 0).astype(np.float32)
    
    return input_tensor, (orig_h, orig_w), scale


def main():
    from immich_ml.sessions.cix import CixSession

    # Test image path
    test_image = Path("/mnt/tank/media/photos/Sony J9210/100ANDRO/DSC_0427.JPG")

    if not test_image.exists():
        print(f"Test image not found: {test_image}")
        return

    print(f"Testing with image: {test_image}")
    
    # Load model
    cache = Path.home() / ".cache/immich_ml"
    model_path = cache / "facial-recognition/buffalo_l/detection/cix/model.cix"
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    session = CixSession(model_path)
    
    # Load and preprocess image
    image = Image.open(test_image).convert("RGB")
    input_tensor, orig_size, scale = preprocess_image(image)
    
    print(f"Original size: {orig_size}, scale: {scale:.3f}")
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Run inference
    import time
    start = time.perf_counter()
    outputs = session.run(None, {"input.1": input_tensor})
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"Inference time: {elapsed:.1f}ms")
    print(f"Output shapes: {[o.shape for o in outputs]}")
    
    # Decode outputs
    result = decode_scrfd_outputs(
        outputs,
        input_size=640,
        orig_size=orig_size,
        scale=scale,
        score_threshold=0.5,
        iou_threshold=0.4,
    )
    
    print(f"\nDetected {len(result['scores'])} faces:")
    for i, (box, score, kps) in enumerate(zip(result['boxes'], result['scores'], result['landmarks'])):
        print(f"  Face {i+1}: score={score:.3f}, box={box.astype(int).tolist()}")
        print(f"           landmarks={kps.astype(int).tolist()}")


if __name__ == "__main__":
    main()
