# immich-cix：Immich on CIX P1 (Radxa Orion O6)

这个目录记录本 fork 在上游 Immich 之上做了什么、为什么这样做，以及每次跟进上游（re-patch）时需要知道的事。
上游没有 `cix/` 目录，所以它不会和上游产生冲突。

| | |
|---|---|
| 当前基线 | 上游 **v3.2.4**，分支 `cix-v3.2.4`（2026-10-06 起用于生产） |
| 硬件 | CIX P1 SoC：V4L2M2M VPU（`mvx` 驱动）、周易 NPU（`libnoe` 3.0）、Immortalis G720 GPU |
| 部署方式 | 裸机：systemd + rootless podman 的 PostgreSQL + 本机 Redis；**不是** Docker |

## 补丁清单

| 领域 | 改了什么 | 主要文件 | 能否上游化 |
|---|---|---|---|
| V4L2M2M 转码 | 新的 `v4l2m2m` 硬件加速类型（软解/硬解 + 硬编），绕开 CIX ffmpeg 的若干坑 | `server/src/utils/media.ts`、`enum.ts`、`constants.ts`（`SUPPORTED_HWA_CODECS`）、`FFmpegSettings.svelte`、`i18n/en.json`、OpenAPI + SDK | 通用部分可以（需去掉 CIX 特有的 `dslw` 处理或做成可选） |
| tonemapx 回退 | ffmpeg 没有 jellyfin 的 `tonemapx` 时改用 `zscale + tonemap` | `media.ts`（`setTonemapxAvailable`）、`media.service.ts`、`media.repository.ts`（`hasFilter`） | 可以（对所有非 jellyfin-ffmpeg 的裸机部署都有用） |
| CIX NPU 推理 | `CixSession`：libnoe 封装、量化/反量化、按张量类型映射 dtype、每 session 加锁 | `machine-learning/immich_ml/sessions/cix/`、`schemas.py`（`ModelFormat.CIX`）、`config.py`（`cix`）、`models/base.py` | 需要等上游接受第三方 NPU 后端；v3.3 的 session 协议重写后再评估 |
| Chinese-CLIP | 注册 `chinese-clip-vit-{base-patch16,large-patch14}`；open_clip `resize_mode: squash` | `models/constants.py`、`models/clip/visual.py`、`server/src/constants.ts`（`CLIP_MODEL_INFO`） | `resize_mode` 可以；模型本身依赖手工部署的 `.cix` |
| 测试 | V4L2M2M / tonemapx / CixSession 单测；ML 测试在有 NPU 的机器上保持封闭 | `media.service.spec.ts`、`machine-learning/test_main.py`、`conftest.py` | 随功能一起 |
| 部署脚本 | 把 CIX Model Hub 的 `.cix` 放进 Immich 模型缓存 | `machine-learning/scripts/deploy_*.sh`、`make_chinese_clip_tokenizer.py` | 否 |

**遗留文件**（不再使用，可在下次整理补丁时删除以减少 rebase 噪音）：
`machine-learning/scripts/run_cix_ml_server.py`（旧的独立 ML 服务，约 960 行）及 `test_scrfd_postprocess.py`、`test_ocr*.py`、`test_cix_*.py`、`test_official_pipeline.py`。
它们占了 fork 差异的大部分行数，但生产已改用官方 `immich_ml` pipeline。

## 文档索引

- [REPATCH.md](REPATCH.md) — **跟进上游的操作手册**：步骤、冲突热点、git 不会报的语义适配清单、验证与切换、回滚
- [findings/ffmpeg-vpu.md](findings/ffmpeg-vpu.md) — CIX ffmpeg / VPU 的坑（`-qp`、`dslw`、`tonemapx`、jellyfin-ffmpeg 不可用……）
- [findings/npu-ml.md](findings/npu-ml.md) — NPU / libnoe / immich_ml 的坑（dtype、线程安全、人脸/OCR 管线变化、v3.3 重写）
- [findings/deployment.md](findings/deployment.md) — 裸机部署的坑（备份、pasta、构建工具链、迁移、升级后必做的任务）
- [findings/model-hub.md](findings/model-hub.md) — CIX AI Model Hub 与 cix-prebuilt 发布的笔记
- [models/chinese-clip.md](models/chinese-clip.md) — Chinese-CLIP 的集成方式与评估数据
- [upgrade-logs/](upgrade-logs/) — 每次升级的时间线、决策与测量数据
- [tools/](tools/README.md) — 可复用的验证脚本（演练环境、ML 对比、ffmpeg 检查）
