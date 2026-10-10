#!/bin/bash
# Deploy CIX NPU models for Immich ML
# This script copies pre-quantized models from CIX AI Model Hub to Immich cache

set -e

# Source paths (CIX AI Model Hub)
CIX_MODEL_HUB="/home/radxa/.cache/modelscope/hub/models/cix/ai_model_hub_25_Q3"
CLIP_DIR="$CIX_MODEL_HUB/models/Generative_AI/Image_to_Text/onnx_clip"
FACE_DIR="$CIX_MODEL_HUB/models/ComputeVision/Face_Recognition/onnx_scrfd_arcface"

# Target paths (Immich ML cache)
IMMICH_CACHE="${MACHINE_LEARNING_CACHE_FOLDER:-$HOME/.cache/immich_ml}"
# must match the ML service: any revision but "main" adds a directory level
REVISION="${MACHINE_LEARNING_MODEL_REVISION:-main}"
SUFFIX=""
if [ "$REVISION" != main ]; then SUFFIX="/$REVISION"; fi
CLIP_CACHE="$IMMICH_CACHE/clip/ViT-B-32__openai$SUFFIX"
FACE_CACHE="$IMMICH_CACHE/facial-recognition/buffalo_l$SUFFIX"

echo "=== CIX NPU Model Deployment for Immich ML ==="
echo ""

# Check source models exist
if [ ! -f "$CLIP_DIR/clip_visual.cix" ]; then
    echo "ERROR: CLIP visual model not found at $CLIP_DIR/clip_visual.cix"
    echo "Please download CIX AI Model Hub first:"
    echo "  pip install modelscope"
    echo "  modelscope download --model cix/ai_model_hub_25_Q3"
    exit 1
fi

echo "Source models found:"
echo "  CLIP Visual: $CLIP_DIR/clip_visual.cix ($(du -h "$CLIP_DIR/clip_visual.cix" | cut -f1))"
echo "  CLIP Text:   $CLIP_DIR/clip_txt.cix ($(du -h "$CLIP_DIR/clip_txt.cix" | cut -f1))"
echo "  SCRFD:       $FACE_DIR/scrfd.cix ($(du -h "$FACE_DIR/scrfd.cix" | cut -f1))"
echo "  ArcFace:     $FACE_DIR/arcface.cix ($(du -h "$FACE_DIR/arcface.cix" | cut -f1))"
echo ""

# Create target directories
echo "Creating target directories..."
mkdir -p "$CLIP_CACHE/visual/cix"
mkdir -p "$CLIP_CACHE/textual/cix"
mkdir -p "$FACE_CACHE/detection/cix"
mkdir -p "$FACE_CACHE/recognition/cix"

# Copy CLIP models
echo "Deploying CLIP models..."
cp -v "$CLIP_DIR/clip_visual.cix" "$CLIP_CACHE/visual/cix/model.cix"
cp -v "$CLIP_DIR/clip_txt.cix" "$CLIP_CACHE/textual/cix/model.cix"

# Copy Face models
echo "Deploying Face models..."
cp -v "$FACE_DIR/scrfd.cix" "$FACE_CACHE/detection/cix/model.cix"
cp -v "$FACE_DIR/arcface.cix" "$FACE_CACHE/recognition/cix/model.cix"

# Also copy config files for CLIP (needed by Immich)
echo "Copying config files..."
if [ -d "$CIX_MODEL_HUB/models/Generative_AI/Image_to_Text/onnx_clip/model" ]; then
    # Check if there are any config files
    find "$CIX_MODEL_HUB/models/Generative_AI/Image_to_Text/onnx_clip" -name "*.json" -exec cp -v {} "$CLIP_CACHE/" \; 2>/dev/null || true
fi

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Model locations:"
echo "  $CLIP_CACHE/visual/cix/model.cix"
echo "  $CLIP_CACHE/textual/cix/model.cix"
echo "  $FACE_CACHE/detection/cix/model.cix"
echo "  $FACE_CACHE/recognition/cix/model.cix"
echo ""
echo "Total size:"
du -sh "$IMMICH_CACHE"
echo ""
echo "To use CIX NPU acceleration, ensure:"
echo "  1. libnoe Python package is installed"
echo "  2. CIX NPU driver is loaded (/dev/aipu exists)"
echo "  3. MACHINE_LEARNING_CIX=true (default)"
