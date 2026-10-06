#!/bin/bash
# Deploy Chinese-CLIP (OFA-Sys) for the CIX NPU into the Immich ML model cache.
#
# Usage: deploy_chinese_clip.sh [base|large]
#   base  -> chinese-clip-vit-base-patch16  (512-d, ~41 ms/image on the NPU)
#   large -> chinese-clip-vit-large-patch14 (768-d, ~190 ms/image on the NPU)
#
# The .cix files come from the CIX AI Model Hub (downloaded with `modelscope download` if missing; keep the hub on a
# disk with space, e.g. HUB_DIR=/mnt/tank/...). The tokenizer.json reproduces cn_clip.tokenize (BERT WordPiece, [CLS]
# ... [SEP], padded with 0 to 52 tokens); it is built from cn_clip's vocab.txt.

set -euo pipefail

VARIANT="${1:-base}"
HUB_DIR="${HUB_DIR:-/mnt/tank/cix-model-hub/ms_26_Q2}"
CACHE_DIR="${MACHINE_LEARNING_CACHE_FOLDER:-$HOME/.cache/immich_ml}"
PYTHON="${PYTHON:-$(dirname "$0")/../.venv/bin/python}"
VOCAB="${VOCAB:-}"
HUB_PREFIX="models/Generative_AI/Image_to_Text"

case "$VARIANT" in
  base)
    NAME=chinese-clip-vit-base-patch16 DIM=512 PATCH=16
    SRC="$HUB_PREFIX/onnx-Chinese-clip-vit-base-patch16" IMG=image_encoder.cix TXT=text_encoder.cix ;;
  large)
    NAME=chinese-clip-vit-large-patch14 DIM=768 PATCH=14
    SRC="$HUB_PREFIX/onnx_Chinese_clip" IMG=clip_cn_img.cix TXT=clip_cn_txt.cix ;;
  *) echo "usage: $0 [base|large]" >&2; exit 1 ;;
esac

if [ ! -f "$HUB_DIR/$SRC/$IMG" ] || [ ! -f "$HUB_DIR/$SRC/$TXT" ]; then
  echo "Downloading $SRC/*.cix from cix/ai_model_hub_26_Q2 to $HUB_DIR"
  modelscope download --model cix/ai_model_hub_26_Q2 --include "$SRC/$IMG" "$SRC/$TXT" --local_dir "$HUB_DIR"
fi

if [ -z "$VOCAB" ]; then
  VOCAB=$(python3 -c "import cn_clip, os; print(os.path.join(os.path.dirname(cn_clip.clip.__file__), 'vocab.txt'))" 2>/dev/null || true)
fi
if [ -z "$VOCAB" ] || [ ! -f "$VOCAB" ]; then
  VOCAB=$(mktemp --suffix=-vocab.txt)
  curl -fsSL -o "$VOCAB" "https://huggingface.co/OFA-Sys/$NAME/resolve/main/vocab.txt"
fi

DEST="$CACHE_DIR/clip/$NAME"
mkdir -p "$DEST/visual/cix" "$DEST/textual/cix"
cp "$HUB_DIR/$SRC/$IMG" "$DEST/visual/cix/model.cix"
cp "$HUB_DIR/$SRC/$TXT" "$DEST/textual/cix/model.cix"
cat > "$DEST/config.json" <<JSON
{"embed_dim": $DIM, "vision_cfg": {"image_size": 224, "patch_size": $PATCH}, "text_cfg": {"context_length": 52, "vocab_size": 21128}}
JSON
cat > "$DEST/visual/preprocess_cfg.json" <<'JSON'
{"size": [224, 224], "mode": "RGB", "mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711], "interpolation": "bicubic", "resize_mode": "squash"}
JSON
echo '{"pad_token": "[PAD]"}' > "$DEST/textual/tokenizer_config.json"
"$PYTHON" "$(dirname "$0")/make_chinese_clip_tokenizer.py" "$VOCAB" "$DEST/textual/tokenizer.json"

echo "Deployed $NAME to $DEST"
echo "Then set Administration > Machine Learning > Smart Search > CLIP model to '$NAME' and run Smart Search (All)."
