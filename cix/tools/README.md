# 验证工具

都是 2026-10-06 升级时实际用过的脚本整理而来。数据库只读访问通过 `podman exec $PG_CONTAINER psql`
（环境变量 `PG_CONTAINER`/`PG_PORT`/`PG_USER`/`PG_DB`，默认 `immich_postgres`/5432/postgres/immich）。

## dry-run/ —— 演练环境

| 脚本 | 作用 |
|---|---|
| `start.sh` | 用生产数据库的副本 + overlay 只读媒体库 + Redis DB 1 起一套新版本（server :2284，ML :3013，DB :5433） |
| `stop.sh` | 停掉（保留数据；删掉 `$WORK` 和容器 `immich_postgres_dryrun` 即重置） |

```bash
WORKTREE=/home/radxa/immich-cix/immich-vX BUILD_DATA=/home/radxa/immich-build-vX \
MEDIA_LOCATION=<生产的 IMMICH_MEDIA_LOCATION> DUMP=<pg_dump -Fc 文件> ./dry-run/start.sh
```

不要让演练环境过夜：夜间的完整性检查会读整个 HDD。

## ffmpeg/

| 脚本 | 作用 |
|---|---|
| `check_v4l2m2m.sh [ffmpeg]` | 检查 CIX ffmpeg 的已知坑是否仍在：滤镜（`tonemapx`/`zscale`/…）、编解码器、`-qp` 与 `-rc_enable`、`dslw` 与 `scale=-2`；附码率和 PSNR。每次换 ffmpeg 包都跑一次 |

## ml/

在 `machine-learning/` 目录下用 venv 运行（需要 `immich_ml` 的脚本要设 `PYTHONPATH=$PWD`、`LD_LIBRARY_PATH=/usr/share/cix/lib`、`HF_HUB_OFFLINE=1`，以及指向模型所在缓存的 `MACHINE_LEARNING_CACHE_FOLDER`）。

| 脚本 | 作用 |
|---|---|
| `compare_servers.py A B img...` | 对两个 ML 服务发同样的请求，比较 CLIP 向量（cos）、人脸框（IoU）和人脸向量 |
| `embeddings_vs_db.py clip\|faces URL` | 重新计算并和数据库里存的向量比；默认固定样本，便于跨版本对比 |
| `ocr_check.py URL` | OCR 新结果和数据库里的文字并排打印 |
| `cix_inspect.py model.cix...` | 打印 `.cix` 的输入/输出张量类型、大小、scale、zero point（只加载图，不推理） |
| `clip_npu_vs_onnx.py MODEL` | 同一模型 NPU（走 Immich 自己的预处理/分词）对 fp32 ONNX 的 cos 和延迟 |
| `clip_eval.py sample\|embed\|query` | 在真实图库样本上比较多个 CLIP 模型的检索结果，输出对照图 |

例：

```bash
cd /home/radxa/immich-cix/immich-vX/machine-learning
T=../cix/tools/ml
.venv/bin/python $T/compare_servers.py http://127.0.0.1:3003 http://127.0.0.1:3013 <几张 preview.jpeg>
.venv/bin/python $T/embeddings_vs_db.py faces http://127.0.0.1:3013 --limit 40

export HF_HUB_OFFLINE=1 LD_LIBRARY_PATH=/usr/share/cix/lib PYTHONPATH=$PWD MACHINE_LEARNING_CACHE_FOLDER=$HOME/.cache/immich_ml
.venv/bin/python $T/clip_eval.py sample /tmp/eval --n 1500
.venv/bin/python $T/clip_eval.py embed /tmp/eval chinese-clip-vit-large-patch14
.venv/bin/python $T/clip_eval.py query /tmp/eval --models chinese-clip-vit-large-patch14 --queries "海边日落,猫,火锅"
```
