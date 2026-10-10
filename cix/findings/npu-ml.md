# NPU / libnoe / immich_ml 的坑

验证日期 2026-10-06：`libnoe` 3.0.0、`cix-npu-driver` 3.0.0、Python 3.11.13、上游 immich_ml v3.2.4。

## libnoe 3.0 API

- 3.0 返回 `(status, data)` 元组（旧版是 `{'ret': …, 'data': …}` 字典），`noe_get_tensor(job, type, idx)` 返回 `bytes`，`noe_create_job` 需要 `noe_create_job_cfg_t()`。`CixSession._unpack()` 两种都兼容
- wheel 在 `/usr/share/cix/pypi/libnoe-3.0.0-py3-none-manylinux2014_aarch64.whl`，内含 cp310–cp313 的 `.so`，所以 Python 3.12/3.13 也能用（`onnxruntime_zhouyi` 只有 cp311，但我们不用它）
- 运行时需要 `LD_LIBRARY_PATH=/usr/share/cix/lib`
- 张量描述符只有 `data_type, id, scale, size, zero_point`，**没有形状**：人脸/OCR 用静态形状表，CLIP 按 `size / itemsize` 推导
- 量化 `q = round(x * scale - zero_point)`，反量化 `x = (q + zero_point) / scale`

## 张量类型因模型而异

| 模型 | 输入 | 输出 |
|---|---|---|
| CLIP ViT-B-32（旧） | 图 S8 / 文本 S32×77 | S8×512 |
| Chinese-CLIP B/16 | 图 S8 / 文本 S32×52 | **S16**×512 |
| Chinese-CLIP L/14 | 图 S8 / 文本 S32×52 | 图 S8×768 / 文本 **F16**×768 |
| SCRFD | S8 | scores **U8**，bbox/kps S8 |

旧 `CixSession` 把所有非 U8 输出当 int8 读，Chinese-CLIP 的输出会被读成垃圾。现在按描述符类型映射 numpy dtype（S8/U8/S16/U16/S32/U32/S64/U64/F16/F32），浮点类型不反量化；只有"浮点数据 → 整数张量"才量化，token id 之类的整数输入原样传入。
查看任何 `.cix` 的描述符：[`tools/ml/cix_inspect.py`](../tools/ml/cix_inspect.py)。

## 线程安全

immich_ml 在 12 线程的线程池里跑推理，Smart Search 默认并发 2；而一个 `CixSession` 只有一个 NPU job。
压测（8 线程、同一 session）：**不加锁 33/64 个结果错误**，加锁后 0/64。现在 `run()` 持有每个 session 的锁。
生产在 8 月切到官方 pipeline 后没有新上传，所以之前没暴露；全量重建索引正是会触发它的场景。

## `is_available` 在 import 时探测真实硬件

`sessions/cix/__init__.py` 在导入时就打开一次 NPU context。后果：在板子上跑上游测试时，默认模型格式变成 CIX，十几个上游测试失败。
`conftest.py` 里的自动 fixture `cix_unavailable` 默认把它 mock 掉；需要 CIX 的测试自己再 patch 成 True。

## v3.2 的 ML 管线变化（相对 v2.6）

- **去掉了 insightface**：SCRFD 解码 / NMS / 对齐在 `models/facial_recognition/_ops.py`。它要 9 个头 `[scores×3, boxes×3, kps×3]`、每格 2 个 anchor、步长 8/16/32，`ensure_dims` 会把我们的 2-D 输出补成 4-D，直接兼容
- 预处理：检测 letterbox + RGB、mean 127.5 / std 128；识别 RGB、127.5 / 127.5 —— 与 CIX hub 的参考实现一致
- `FaceRecognizer` 对非 ONNX 格式 batch=1，不会走 `onnx.load()` 加 batch 轴，所以旧的 `_CixArcFaceWrapper` 不再需要
- **OCR 写死了 ONNX Runtime**（`# TODO: support other runtimes`）：`PP-OCRv4-cix` 在 v3.2 上会被当成名字里没有 "mobile" 的模型，下载 PP-OCRv5 **server** 版并在 CPU 上跑（11–13 s/图）。该选项已删除
- 旧的 "StrEnum + match/case 在 3.11 上不匹配" workaround 是不必要的：上游 `models/__init__.py` 在 3.11 上对枚举和字符串都能正确分派，已恢复成上游版本

## 人脸

- 部署的模型其实是 **det_10g + w600k_r50**（和上游 buffalo_l 同一对），不是 SCRFD-500m / MobileFaceNet：build 配置写的是 `det_10g.onnx` / `w600k_r50.onnx`，层名比对也吻合
- 新管线与数据库里已存 embedding 的余弦：中位数 0.91–0.95，最低 0.65–0.85（库里大多是旧独立服务算的，旧服务还有把 BGR 喂给 ArcFace 的 bug）。Immich 聚类阈值是距离 0.5，所以**不需要重跑人脸识别**
- 对比要用**固定样本**（`embeddings_vs_db.py faces` 默认按 id 排序）：随机样本之间差异很大，上次就因此误以为有回归

## OCR

- 旧 PP-OCRv4 NPU 管线的结果大多是错的：碎片、错字、无字图片上报出 "1"、"0" 之类的文字
- PP-OCRv5_mobile 在 CPU（ORT）上约 1–2 s/图，准确，无字图片正确返回空。生产已切换并全量重跑
- 若以后要回到 NPU：v3.3 的识别输入高度固定 48（CIX v4 模型是 32×400）、检测归一化是 `x/127.5-1`（CIX 模型按 ImageNet mean/std 校准）、需要 `charset.txt`、输出要叫 `ctc_logits`

## 运维细节

- `python -m immich_ml` 会用 PATH 里的 `python` 启动 gunicorn：**venv 的 bin 必须在 PATH 最前**，否则会用到系统 Python
- 停 `immich_ml` 启动器不会带走 gunicorn 子进程；按端口匹配杀（`pkill -f '[-]b 127.0.0.1:3013'`，`[-]` 防止匹配到执行它的 shell 自己）
- 模型 5 分钟没用会卸载（`MACHINE_LEARNING_MODEL_TTL` 默认 300），之后的首个请求要重新加载（Chinese-CLIP L/14 实测首个搜索约 360 ms，之后 80–140 ms）
- server 重启时会立刻恢复被中断的任务；如果 ML 连接还没初始化，会在第一秒内报 "Machine learning repository not been setup"（2026-10-06 那次有 30 个 OCR 任务因此失败，之后跑一次 OCR Missing 即可）

## v3.3 的 ML 重写

v3.3.0-rc.0 起（3.3.0 正式版 2026-10-06 发布，3.3.1 在 10-08）：

- `requires-python >= 3.12`（PEP 695 泛型语法，3.11 直接 SyntaxError）；新增依赖 `immich-model`
- `models/__init__.py` 清空，分派改到 `pipeline.py`（严格校验请求）；`models/cache.py` 按 `(class, name, options)` 缓存
- session 协议：session 提供 `shapes`、`batches`、`for_shape(shape) -> graph`、`warm()`；graph 提供 `run`、`get_inputs`、`get_outputs`、`get_metadata`、`normalizes_input`（True 时模型收原始 uint8 NHWC）
- `sessions/policy.py`（ShapePolicy）、`sessions/prepare.py`（只针对 ORT 的子进程准备）
- 人脸/OCR 的后处理已全部内置（`_ops.py`、`ocr/postprocess.py`、`ocr/ctc.py`）

### 移植结果（2026-10-10，`cix-v3.3.1`）

- `CixSession` 照 `AnnSession` 的写法同时充当 session 和 graph：`shapes=(Shape(batch=1),)`、`batches=(1,)`、`for_shape` 返回自身、`warm` 空操作、`get_metadata` 返回 `{}`、`normalizes_input=False`（模型收归一化后的浮点，`run()` 里量化），锁保留
- `base.py`：`_IGNORED_PATTERNS[CIX]`、`_make_session` 的 `case ModelFormat.CIX`、`model_path` → `cix.model_path(model_dir)` = `<model_dir>/cix/model.cix`
- **默认格式改为"有 `.cix` 才用 CIX"**：`.cix` 从不下载，所以 OCR、没编译过的 CLIP 模型等直接走 ONNX，不再先下载 HF 仓库、抛 `FileNotFoundError` 再靠 `main.py` 回退
- 删掉了 PP-OCRv4 的张量表和 insightface 的 `set_providers()`
- 人脸识别 `batches=(1,)` 时每张脸单独跑一次（`runs(n, (1,))`），和以前一样
- CLIP 文本按 `get_inputs()` 的名字取 token（`{node.name: tokens[node.name]}`），所以文本模型第一个输入必须叫 `text`（张量表里就是这么写的）
- 板上验证（同一批 preview 图，v3.2.4 生产 vs v3.3.1）：Chinese-CLIP L/14 文本/图像 cos = 1.00000；人脸框 IoU = 1.000、分数一致、embedding cos = 1.0000；OCR（CPU）与数据库逐字一致；16 线程 × 32 个请求与串行结果完全一致

### 模型 revision（`MACHINE_LEARNING_MODEL_REVISION`）

- 默认 `main` = 旧导出（`legacy_models`），缓存路径不变：`<cache>/<task>/<model>/<type>/cix/model.cix`
- 设成 `v2`（上游 main 的 compose 已默认设上，以后会成为默认值）后缓存多一层：`<cache>/<task>/<model>/v2/…`，**手工部署的 `.cix` 和 CLIP 配置也得放进 `v2/`**（两个部署脚本会跟随这个环境变量）
- v2 只影响走 ORT 的模型（对我们是 CPU 上的 OCR）：优化后的导出更快、内存更少，输出数值等价；`models/ocr/legacy.py` 在 `legacy_models` 时强制 ONNX + rapidocr 下载
- 还没试：v2 下的 OCR 速度、PP-OCRv6（tiny/small/medium，全部多语言）
