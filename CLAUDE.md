# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork of Immich with custom hardware acceleration support for CIX P1 SoC (Orion O6) using V4L2M2M (Video4Linux2 Memory-to-Memory). The goal is to enable hardware-accelerated video transcoding and potentially hardware-accelerated machine learning on this ARM-based SBC.

**Key technologies:**
- Backend: Node.js, NestJS, TypeScript
- Media processing: FFmpeg with V4L2M2M hardware acceleration
- Machine Learning: ONNX Runtime (potential CIX NPU support via `onnxruntime_zhouyi`)

## Repository Structure

```
immich-cix/
├── immich-cix/           # Main Immich fork
│   ├── server/           # Backend server (NestJS)
│   │   └── src/
│   │       ├── enum.ts                    # TranscodeHardwareAcceleration enum
│   │       ├── utils/media.ts             # V4L2M2M codec configurations
│   │       └── dtos/system-config.dto.ts  # FFmpeg configuration DTOs
│   ├── machine-learning/ # ML service (Python, ONNX Runtime)
│   ├── web/              # Frontend (Svelte)
│   └── docs/             # Documentation
│
├── ffmpeg-cix/           # CIX-optimized FFmpeg fork
│   └── libavcodec/       # V4L2M2M encoder/decoder implementations
│
└── orion-docs/           # Orion O6 / CIX P1 documentation
    └── o6/app-development/artificial-intelligence/  # NPU documentation
```

---

## CIX P1 SoC Hardware Capabilities

### Video Processing Unit (VPU)

The CIX P1 SoC provides V4L2M2M-based hardware video encoding/decoding:

| Device | Type | Supported Formats |
|--------|------|-------------------|
| `/dev/video3` | Decoder | H.264, HEVC, VP8, VP9, AV1, MPEG-2, MPEG-4, VC-1, JPEG, MJPEG |
| `/dev/video4` | Encoder | H.264, HEVC, VP8, VP9, JPEG, MJPEG |

**Symbolic links:**
- `/dev/video-cixdec0` → `/dev/video3`

### GPU (Mali)

- Device: `/dev/mali0`
- Library: `/opt/cixgpu-pro/lib/aarch64-linux-gnu/libmali.so`
- Supports OpenCL for tonemapping

### NPU (Zhouyi/周易)

- Driver: `cix-npu-driver`
- ONNX Runtime: `/usr/share/cix/pypi/onnxruntime_zhouyi-1.19.0-cp311-cp311-linux_aarch64.whl`
- Note: Currently not integrated with Immich ML

---

## V4L2M2M Hardware Transcoding Implementation

### Key Files Modified

1. **`server/src/enum.ts`** - Added `V4l2m2m = 'v4l2m2m'` to `TranscodeHardwareAcceleration` enum

2. **`server/src/utils/media.ts`** - Added V4L2M2M configuration classes:
   - `V4l2m2mSwDecodeConfig` - Software decode + hardware encode
   - `V4l2m2mHwDecodeConfig` - Hardware decode + hardware encode

3. **`web/src/lib/components/admin-settings/FFmpegSettings.svelte`** - Added V4L2M2M option to hardware acceleration dropdown

4. **`i18n/en.json`** - Added translation: `"transcoding_acceleration_v4l2m2m": "V4L2M2M (for ARM SoCs like CIX P1)"`

5. **`open-api/immich-openapi-specs.json`** - Regenerated to include `v4l2m2m` in `TranscodeHWAccel` enum

6. **`open-api/typescript-sdk/`** - Rebuilt SDK with V4L2M2M support

### Critical Implementation Notes

#### V4L2M2M Does NOT Require Device Initialization
Unlike VAAPI/QSV/NVENC, V4L2M2M doesn't need `-init_hw_device` or `-hwaccel_device` flags.
Auto-detection happens when decoder/encoder is specified with `-c:v`.

```typescript
getBaseInputOptions(): string[] {
  return [];  // No device init needed
}
```

#### V4L2M2M Encoders Do NOT Support `-level` Parameter
Attempting to pass `-level` causes FFmpeg to fail with "Error setting option level".

```typescript
getPresetOptions() {
  return [];  // No -level parameter
}
```

#### Pixel Format Requirements
- **Video encoders** (h264, hevc): Require `yuv420p` format

```typescript
// Always ensure yuv420p for V4L2M2M video encoder
options.push('format=yuv420p');
```

#### Supported Hardware Decoders

```typescript
private static readonly V4L2M2M_HW_DECODERS: Record<string, string> = {
  h264: 'h264_v4l2m2m',
  hevc: 'hevc_v4l2m2m',
  vp8: 'vp8_v4l2m2m',
  vp9: 'vp9_v4l2m2m',
  av1: 'av1_v4l2m2m',
  mpeg2video: 'mpeg2_v4l2m2m',
  mpeg4: 'mpeg4_v4l2m2m',
};
```

#### AV1 10-bit Decoding Limitation

**Warning**: CIX SoC's `av1_v4l2m2m` decoder does NOT support 10-bit AV1 videos (`yuv420p10le`).
Attempting to decode 10-bit AV1 will cause the decoder to enter an infinite `POLLERR` loop.

**Recommendation**: Use software decoder `libdav1d` for 10-bit AV1 content.

### HDR Tonemapping (V4L2M2M + OpenCL)

When Mali GPU is available, HDR to SDR tonemapping can be done via OpenCL:

```
Input (HDR) → V4L2M2M decoder → scale → hwupload →
tonemap_opencl → hwdownload → format=yuv420p → V4L2M2M encoder → Output (SDR)
```

Supported tonemapping algorithms: `hable`, `mobius`, `reinhard`
**NOT supported**: `bt2390` (will silently disable tonemapping)

---

## Development Environment

### Prerequisites

- Node.js 24.x
- pnpm 10.x (`npm install -g pnpm@10`)
- PostgreSQL (via podman/docker)
- Redis (via podman/docker or native)

### Quick Setup

```bash
cd /home/radxa/immich-cix/immich-cix

# 1. Install pnpm if not installed
npm install -g pnpm@10

# 2. Create .env file
cp docker/example.env docker/.env
# Edit docker/.env to set UPLOAD_LOCATION

# 3. Install dependencies
pnpm install --filter '!documentation'

# 4. Build server and SDK
make build-server
make build-sdk
```

### Running Development Server

#### Option 1: Docker Development (Official)

```bash
cd /home/radxa/immich-cix/immich-cix
make dev
```

Access:
- Web UI: http://localhost:3000
- API: http://localhost:2283

For V4L2M2M hardware testing, add to `docker/docker-compose.dev.yml`:
```yaml
devices:
  - /dev/video3:/dev/video3
  - /dev/video4:/dev/video4
```

#### Option 2: Local Development (Recommended for Hardware Testing)

```bash
# Start database and redis
podman start immich_postgres

# Set environment variables
export DB_HOSTNAME=localhost
export DB_PORT=5432
export DB_USERNAME=postgres
export DB_PASSWORD=postgres
export DB_DATABASE_NAME=immich
export REDIS_HOSTNAME=localhost
export IMMICH_MACHINE_LEARNING_URL=http://localhost:3003
export IMMICH_MEDIA_LOCATION=/data/wwwroot/im.ripplecraft.cn/library
export IMMICH_BUILD_DATA=/tmp/immich-build

# Prepare build directory (first time only)
mkdir -p /tmp/immich-build/corePlugin
cp -r plugins/manifest.json plugins/dist /tmp/immich-build/corePlugin/
# Copy geodata from container if needed:
# podman cp immich_server:/build/geodata /tmp/immich-build/

# Start server (with hot-reload)
cd /home/radxa/immich-cix/immich-cix/server
npm run start:dev

# Start web (in another terminal)
cd /home/radxa/immich-cix/immich-cix/web
IMMICH_SERVER_URL=http://localhost:2283 pnpm run dev
```

**Note:** The plugins must be built first with `cd plugins && npm run build` (requires `extism-js` CLI tool).

### Development Commands

```bash
# Type check
cd server && npm run check

# Run unit tests
cd server && pnpm test

# Format code
make format-server

# Lint code
make lint-server

# Build all components
make build-all

# Regenerate OpenAPI specs and SDK (after modifying enums/DTOs)
cd open-api && bash ./bin/generate-open-api.sh
make build-sdk
```

### Documentation

See `immich-cix/docs/docs/developer/` for detailed guides:
- `setup.md` - Environment setup
- `testing.md` - Running tests
- `troubleshooting.md` - Common issues

---

## Build Commands

### Server

```bash
cd immich-cix/server
npm install
npm run build        # Development build
npx tsc --noEmit     # Type check only
```

### Testing V4L2M2M Manually

```bash
# Test software decode + hardware encode
ffmpeg -i input.mp4 -vf "format=yuv420p" -c:v hevc_v4l2m2m -f null -

# Test hardware decode + hardware encode
ffmpeg -c:v h264_v4l2m2m -i input.mp4 -vf "format=yuv420p" -c:v hevc_v4l2m2m -f null -

# Test with scaling
ffmpeg -i input.mp4 -vf "scale=1280:720,format=yuv420p" -c:v h264_v4l2m2m output.mp4
```

---

## Configuration

In Immich Web UI:
1. Navigate to **Administration → Video Transcoding Settings**
2. Set **Hardware Acceleration** to **V4L2M2M**
3. Optionally enable **Hardware Decoding**

---

## Known Limitations

1. **Concurrent Transcoding**: Running multiple V4L2M2M tasks concurrently may exhaust hardware resources and cause POLLERR errors

2. **No VP9/AV1 Encoding**: CIX hardware encoder only supports H.264 and HEVC output

3. **Quality Parameter**: V4L2M2M uses `-qp` instead of `-crf` for quality control

---

## Machine Learning Hardware Acceleration

### Overview

CIX P1 has two potential ML acceleration paths:

| 方案 | 硬件 | 数据类型 | Immich 支持 | 状态 |
|------|------|----------|-------------|------|
| ARM NN | Mali GPU | FP32/FP16 | ✅ 官方支持 | ✅ 已验证可用 |
| CIX NPU | 周易 NPU | INT8 | ✅ 自定义集成 | ✅ 全功能可用 |

### Option 1: ARM NN (Mali GPU) - 推荐

**状态：已验证可用**

ARM NN v24.05 已在 CIX P1 上成功测试，可使用 Mali GPU 加速 ML 推理。

**系统要求：**
- `/dev/mali0` - Mali GPU 设备
- `/opt/cixgpu-pro/lib/aarch64-linux-gnu/libmali.so` - Mali 驱动库
- ARM NN v24.05 库 (`libarmnn.so.33`, `libarmnnOnnxParser.so.24` 等)
- `libann.so` - Immich 的 ARM NN 封装库

**部署方式：**
1. 使用官方 Docker 镜像: `ghcr.io/immich-app/immich-machine-learning:release-armnn`
2. 或本地构建 `libann.so` 并配置环境

**构建 libann.so：**
```bash
# 下载 ARM NN
curl -SL "https://github.com/ARM-software/armnn/releases/download/v24.05/ArmNN-linux-aarch64.tar.gz" | tar -zx -C /opt/armnn

# 构建 libann.so
cd immich-cix/machine-learning/ann
export ARMNN_PATH=/opt/armnn
g++ -shared -O3 -o libann.so -fuse-ld=gold -std=c++17 \
    -I"$ARMNN_PATH"/include -L"$ARMNN_PATH" \
    -larmnn -larmnnDeserializer -larmnnTfLiteParser -larmnnOnnxParser \
    ann.cpp
```

**配置 Mali OpenCL：**
```bash
mkdir -p /etc/OpenCL/vendors
echo "/opt/cixgpu-pro/lib/aarch64-linux-gnu/libmali.so" > /etc/OpenCL/vendors/mali.icd
```

**注意：** ARM NN 的 ONNX Parser 将在 v24.08 被移除，未来可能需要使用 TFLite 或 .armnn 格式。

### Option 2: CIX NPU (Zhouyi/周易) - 全功能可用 ✅

**状态：Smart Search + People + OCR 功能完全可用**

CIX NPU 已成功集成到 Immich ML，使用 CIX 官方预量化的 `.cix` 模型文件。

**已完成的工作：**
1. ✅ 创建 `CixSession` 适配器 (`immich_ml/sessions/cix/__init__.py`)
2. ✅ 添加 `ModelFormat.CIX` 枚举到 `schemas.py`
3. ✅ 修改 `base.py` 支持 `.cix` 模型加载
4. ✅ 部署预量化模型到 `~/.cache/immich_ml/`
5. ✅ 创建简化版 ML 服务器 (`scripts/run_cix_ml_server.py`)
6. ✅ 验证 CLIP Visual/Textual 嵌入生成
7. ✅ 实现 SCRFD 后处理（anchor 解码、NMS、人脸对齐）
8. ✅ 集成 ArcFace 人脸识别

**当前性能：**
| 模型 | 推理时间 | 状态 |
|------|----------|------|
| CLIP Visual | ~29ms | ✅ 工作正常 |
| CLIP Textual | ~15ms | ✅ 工作正常 |
| Face Detection (SCRFD) | ~62ms | ✅ 工作正常 |
| Face Recognition (ArcFace) | ~4ms | ✅ 工作正常 |
| **端到端人脸检测+识别** | ~265ms | ✅ 工作正常 |
| OCR Detection (PP-OCRv4) | ~100-155ms | ✅ 工作正常 |
| OCR Recognition (PP-OCRv4) | ~60-80ms/区域 | ✅ 工作正常 |
| **端到端 OCR（2区域）** | ~320ms | ✅ 工作正常 |

**模型文件位置：**
```
~/.cache/immich_ml/
├── clip/ViT-B-32__openai/
│   ├── config.json               # 模型配置
│   ├── visual/cix/model.cix      # CLIP 图像编码器
│   └── textual/
│       ├── cix/model.cix         # CLIP 文本编码器
│       ├── tokenizer.json        # HuggingFace tokenizer (从 immich-app/ViT-B-32__openai 下载)
│       └── tokenizer_config.json
├── facial-recognition/buffalo_l/
│   ├── detection/cix/model.cix   # SCRFD 人脸检测
│   └── recognition/cix/model.cix # ArcFace 人脸识别
└── ocr/PP-OCRv4_mobile/
    ├── detection/cix/model.cix   # PP-OCRv4 文字检测
    ├── recognition/cix/model.cix # PP-OCRv4 文字识别
    └── ppocr_keys_v1.txt         # 字符字典 (6625字符)
```

**模型来源：**
CIX AI Model Hub (ModelScope): `/home/radxa/.cache/modelscope/hub/models/cix/ai_model_hub_25_Q3/`
- `openclip/openclip_vit_b32/` - CLIP ViT-B-32
- `insightface/scrfd_500m_bnkps/` - SCRFD 人脸检测
- `insightface/arcface_mfn/` - ArcFace 人脸识别
- `models/ComputeVision/OCR/onnx_PP_OCRv4/` - PP-OCRv4 OCR

**CixSession 关键实现：**
```python
# immich_ml/sessions/cix/__init__.py
class CixSession:
    def __init__(self, model_path: Path):
        from libnoe import NPU
        self.npu = NPU()
        self.npu.noe_init_context()
        result = self.npu.noe_load_graph(str(model_path))
        self.graph_id = result['data']
        job_result = self.npu.noe_create_job(self.graph_id, {})
        self.job_id = job_result['data']

    def run(self, output_names, input_feed):
        # 量化输入 (float32 → int8)
        # 运行推理
        # 反量化输出 (int8 → float32)
```

**libnoe API 注意事项：**
- `noe_load_graph()` 返回 `{'data': graph_id, 'ret': status}`
- `noe_create_job()` 第二个参数用空字典 `{}`
- `noe_get_tensor()` 需要根据输出类型选择 `D_INT8` 或 `D_UINT8`
- CLIP 文本输入是 int32，不需要量化
- **重要**：SCRFD scores 输出是 U8 类型，需要用 `D_UINT8` 读取

**SCRFD 后处理实现要点：**
```python
# 1. 输出格式：9个张量 [scores_8, scores_16, scores_32, bboxes_8, ..., kps_8, ...]
# 2. Anchor 网格生成
anchor_centers = np.stack(np.mgrid[:feat_h, :feat_w][::-1], axis=-1) * stride
anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))

# 3. 距离解码为边界框
x1 = anchor_centers[:, 0] - bbox_preds[:, 0]
y1 = anchor_centers[:, 1] - bbox_preds[:, 1]
x2 = anchor_centers[:, 0] + bbox_preds[:, 2]
y2 = anchor_centers[:, 1] + bbox_preds[:, 3]

# 4. NMS 去重
# 5. 人脸对齐：5点 landmarks → SimilarityTransform → 112x112
```

**OCR 模型配置：**
- CIX NPU 使用 PP-OCRv4 模型（Immich 默认使用 PP-OCRv5）
- 在 UI 中选择 `PP-OCRv4-cix (CIX NPU, Chinese and English)` 模型
- 支持中英文识别
- 如果选择其他 PP-OCRv5 模型，服务器会自动回退到 PP-OCRv4-cix 并输出日志提示

**已知限制：**
- SCRFD 输入固定为 640x640，高分辨率图片会被缩放
- 对于 4K+ 图片中占比很小的人脸，检测效果可能下降
- OCR 检测输入固定为 960x608，大图会被缩放
- OCR 识别时间随文本区域数量线性增长（~60-80ms/区域）

**运行 CIX ML 服务器：**
```bash
cd /home/radxa/immich-cix/immich-cix/machine-learning
python3 scripts/run_cix_ml_server.py
# 服务运行在 http://localhost:3003
```

**测试脚本：**
```bash
# 测试 SCRFD 后处理（独立运行）
python3 scripts/test_scrfd_postprocess.py

# 测试 CIX Session
python3 scripts/test_cix_session.py

# 测试 OCR 功能（独立运行）
python3 scripts/test_ocr.py

# 端到端 API 测试（包括 OCR）
python3 scripts/test_cix_e2e.py
python3 scripts/test_ocr_e2e.py
```

**当前部署状态（2026-01-04 更新）：**
- Immich Server: 本地开发模式 (v2.4.1)，端口 2283
- Web Frontend: 本地开发模式，端口 3000
- PostgreSQL: `podman` 容器 `immich_postgres`，端口 5432
- Redis: 本机原生服务，端口 6379
- CIX ML Server: 本机 Python 进程，端口 3003
- Media Location: `/data/wwwroot/im.ripplecraft.cn/library`
- External Library: `/mnt/tank/media/photos`
- **功能状态**：Smart Search ✅ | People/Face Recognition ✅ | OCR ✅

**已知问题：**
1. ~~Smart Search 返回 "Immich Server Error"~~ - ✅ 已修复（向量格式问题）
2. 部分图片显示 "Error loading image" - 可能是缩略图生成或权限问题
3. ~~People 功能暂不可用~~ - ✅ 已修复（SCRFD 后处理已实现）
4. ~~人脸检测结果为空~~ - ✅ 已修复（2026-01-03）
   - **原因**：Immich Server 同时发送 `detection` 和 `recognition` 请求类型，`recognition` 处理器会覆盖 `detection` 的结果
   - **修复**：在 `run_cix_ml_server.py` 中添加检查，只在响应未设置时初始化空数组

**重要提示：**
- 首次部署或更新 ML 服务器后，需要在 Immich Administration → Jobs 中手动运行 "Smart Search" 和 "Face Detection" 任务
- 如果人脸检测曾经失败，需要重置数据库状态：
  ```sql
  -- 重置人脸识别状态以重新运行任务
  UPDATE asset_job_status SET "facesRecognizedAt" = NULL;
  ```

**相关 SDK 位置：**
- `/home/radxa/cix_noe_sdk_25_q1_release/` - NOE SDK + 用户指南
- `/home/radxa/cix-sdk/component/cix_opensource/cix_unit_test/cix_npu_onnxruntime_py_test/` - Python 示例

---

## References

- Jellyfin V4L2M2M implementation: `/home/radxa/jellyfin-cix/CLAUDE.md`
- CIX NPU documentation: `/home/radxa/immich-cix/orion-docs/o6/app-development/artificial-intelligence/`
- Immich hardware transcoding docs: `/home/radxa/immich-cix/immich-cix/docs/docs/features/hardware-transcoding.md`
