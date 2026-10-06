# CIX AI Model Hub 与 cix-prebuilt 发布

## Model Hub 的版本

| 名称 | 实际内容 | 位置 |
|---|---|---|
| `ai_model_hub_25_Q3` | 现在部署的人脸/OCR 模型的来源（AIPUBuilder 6.1.3407） | 本机已不存在 |
| `ai_model_hub`（本机还有个 `ai_model_hub_Q3` 软链接指向它） | **其实是 2025 Q4**（ReleaseNote 最新条目是 Q4），名字有误导性 | `~/.cache/modelscope/hub/models/cix/` |
| `ai_model_hub_26_Q2` | 2026 Q2，无 LFS 克隆 | `/mnt/tank/cix-model-hub/ai_model_hub_26_Q2` |

26_Q2 的变化以 LLM/VLM 为主（Qwen3.5/3.6、Qwen3-VL、gemma-4 等），新增 Chinese-CLIP ViT-B/16、tinyvit、yolov11 系列等；删了一批老模型。
**与本项目相关的：人脸（SCRFD、ArcFace）和 PP-OCRv4 完全没变**（LFS oid 与部署文件一致）；ViT-B-32 CLIP 在 Q4 重新量化过（文本编码器改为 16 bit），未采用；没有 PP-OCRv5，没有可用的新人脸模型（yolov5_face 只有配置没有模型文件）。

## 下载：git LFS 走不通

ModelScope 的 git-LFS 端点对这些对象返回 404（`Object does not exist on the server`）。用 `modelscope download`（或 HTTP `…/resolve/master/<path>`），并且**放在 `/mnt/tank`**（根分区只剩约 70 GB）：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://www.modelscope.cn/cix/ai_model_hub_26_Q2.git   # 只要目录结构和小文件
modelscope download --model cix/ai_model_hub_26_Q2 \
  --include 'models/Generative_AI/Image_to_Text/onnx_Chinese_clip/*.cix' \
  --local_dir /mnt/tank/cix-model-hub/ms_26_Q2
```

下载后用 `sha256sum` 对照 LFS 指针文件里的 `oid sha256:` 校验。

## 编译器版本与兼容性

`.cix` 里有构建字符串：`strings model.cix | grep AIPUBuilder`。
2025 Q4 / 26_Q2 的模型是 6.1.3753，现有部署是 6.1.3407；6.1.3753 的 Chinese-CLIP 在当前 libnoe 3.0.0 + 驱动 3.0.0 上正常运行。
用 [`tools/ml/cix_inspect.py`](../tools/ml/cix_inspect.py) 看输入输出类型，不要假设都是 int8（见 [npu-ml.md](npu-ml.md#张量类型因模型而异)）。

## Hub 文件 → Immich 模型缓存

Immich 的路径规则：`$MACHINE_LEARNING_CACHE_FOLDER/<task>/<model>/<type>/cix/model.cix`（默认 `~/.cache/immich_ml`）。

| Hub 文件 | Immich 路径 |
|---|---|
| `onnx_scrfd_arcface/scrfd.cix`（det_10g） | `facial-recognition/buffalo_l/detection/cix/model.cix` |
| `onnx_scrfd_arcface/arcface.cix`（w600k_r50） | `facial-recognition/buffalo_l/recognition/cix/model.cix` |
| `onnx_Chinese_clip/clip_cn_img.cix` / `clip_cn_txt.cix` | `clip/chinese-clip-vit-large-patch14/{visual,textual}/cix/model.cix` |
| `onnx-Chinese-clip-vit-base-patch16/image_encoder.cix` / `text_encoder.cix` | `clip/chinese-clip-vit-base-patch16/{visual,textual}/cix/model.cix` |

CLIP 还需要 `config.json`、`visual/preprocess_cfg.json`、`textual/tokenizer.json`、`textual/tokenizer_config.json`；Chinese-CLIP 用 `machine-learning/scripts/deploy_chinese_clip.sh` 一次生成。

## radxa-pkg/cix-prebuilt `26Q2-2607`

- 这个 release 是围绕 **6.6.89 内核**的（带 `linux-image-6.6.89`）；内核模块包（`cix-npu-driver` 3.0.3、`cix-vpu-driver`、`cix-gpu-dkms`）未安装 —— 6.6.89 以前试过并回退了
- 用户态更新：`cix-noe-umd` 3.1.2（现 3.0.0）、`cix-npu-onnxruntime` 1.2.0（现 1.1.0）、`cix-mnn` 1.3.0、`cix-mesa` 25.1.5、ffmpeg 5.1.7。noe-umd 能否配 3.0.0 的内核驱动未验证，暂不升级
- `cix-ffmpeg 5.1.7`：装到 `/usr/share/cix/{bin,lib}`，与系统 ffmpeg 并存；二进制自报 5.1.6-1，行为与现版本完全一致（见 [ffmpeg-vpu.md](ffmpeg-vpu.md)）
- 下载：`gh release download 26Q2-2607 -R radxa-pkg/cix-prebuilt -p '<name>.deb'`；先 `dpkg-deb -x` 解到临时目录用 `LD_LIBRARY_PATH` 测试，再决定装不装
