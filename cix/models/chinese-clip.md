# Chinese-CLIP（智能搜索）

生产自 2026-10-06 起使用 **`chinese-clip-vit-large-patch14`**（OFA-Sys Chinese-CLIP ViT-L/14，768 维），在 NPU 上运行。之前是 `ViT-B-32__openai`（只懂英文）。

## 为什么是 L/14

在生产库里随机抽 1500 张图，同一批查询分别用各模型排序，把 top-5 拼成对照图人工比较（工具：[`tools/ml/clip_eval.py`](../tools/ml/clip_eval.py)）。

| | 中文查询 | 英文查询 | NPU 单图 | NPU 单查询 | 维度 |
|---|---|---|---|---|---|
| ViT-B-32（旧） | ✗ 几乎全是截图/表情包 | ✓ | ~30 ms | ~15 ms | 512 |
| Chinese-CLIP B/16 | △ 噪声大：空白图、二维码、表情包对每个查询都排在前面（fp32 ONNX 也一样，是模型本身的问题） | △ | 41 ms | 20 ms | 512 |
| **Chinese-CLIP L/14** | ✓ 稳定相关（海边日落、猫、火锅、飞机、合影……） | ✓ 与 ViT-B-32 相当 | 189 ms | 21 ms | 768 |

量化损失（NPU vs fp32 ONNX，同一份输入）：

| | 图像 cos | 文本 cos | top-10 与 ONNX 重合 |
|---|---|---|---|
| B/16 | 均值 0.916（最低 0.85） | 均值 0.963 | 0.56（中文）/ 0.40（英文） |
| L/14 | 均值 0.891（最低 0.85） | 均值 0.991（F16 输出） | — |

图像端量化损失不小，但 L/14 的实际检索结果仍然明显最好，所以看对照图比看 cos 更重要。

## 实现

不需要改 Immich 的分词代码，靠配置文件 + 少量通用改动：

- **分词**：Chinese-CLIP 用 BERT WordPiece（`[CLS] … [SEP]`，0 填充到 52）。用 HF `tokenizers` 生成等价的 `tokenizer.json`（`machine-learning/scripts/make_chinese_clip_tokenizer.py`，词表来自 `cn_clip` 包的 `vocab.txt`），`OpenClipTextualEncoder` 直接加载。与 `cn_clip.tokenize` 在 518 条字符串（含中英混排、全角、emoji、繁体、日韩文、超长文本和库里的真实 OCR 文本）上**逐 token 一致**，经过 Immich 的 `clean_text` 后也一致
- **图像预处理**：Chinese-CLIP 是直接拉伸到 224×224（bicubic），不是"短边缩放 + 中心裁剪"。实现为 open_clip 预处理配置里已有的 `resize_mode: "squash"`（`OpenClipVisualEncoder` 原先没实现这个字段）。与 `cn_clip.image_transform(224)` 逐像素一致（最大差 0）
- **NPU**：文本输入 S32×52（不是参考脚本里转的 int64），B/16 输出 S16、L/14 文本输出 F16 —— 需要 `CixSession` 按描述符类型解析（见 [npu-ml.md](../findings/npu-ml.md#张量类型因模型而异)）
- **注册**：ML `_OPENCLIP_MODELS`、server `CLIP_MODEL_INFO`（512 / 768）；Web 的模型名是自由输入框，不用改

## 部署

```bash
machine-learning/scripts/deploy_chinese_clip.sh large        # 或 base；需要时自动 modelscope download 到 /mnt/tank
```

然后在 管理 → 机器学习 → 智能搜索 里把模型名改成 `chinese-clip-vit-large-patch14`，并手动跑 **Smart Search → All**：

- 维度从 512 变 768 时服务器会改 `smart_search.embedding` 列类型并清空；同维度换模型也会清空。**不会自动重建**
- 重建期间智能搜索只覆盖已处理的图片
- 生产实测：约 260 张/分钟（同时在跑 CPU OCR），27k 张约 100 分钟；搜索响应 80–140 ms
- 重复检测依赖 CLIP 向量，会随之重算

## 以后可以试的

- 用重新量化的 L/14（比如提高图像编码器部分层的精度）换取更高的 NPU/ONNX 一致性
- v3.3 移植后确认 `resize_mode`、tokenizer 加载方式是否被上游改动
