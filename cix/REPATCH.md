# 跟进上游（re-patch）操作手册

基于 2026-10-06 从 v2.6.1 跟进到 v3.2.4 的实际经验（过程见 [upgrade-logs](upgrade-logs/)）。
核心原则：**数据库迁移是单向的**，所以一切在 worktree 和数据库副本上先做完、验证完，再动生产。

## 0. 开工前先确认生产环境的真实状态

上次就是在这一步发现了好几个和"以为的状态"不一致的地方：

- [ ] **备份真的在工作**：`library/backups/` 里有最近的 `immich-db-backup-*.sql.gz`（上次发现已经一年多没有成功过，见 [deployment.md](findings/deployment.md#数据库备份)）
- [ ] **生产代码都已提交**：`git status` 干净。ML 服务直接跑 worktree 里的代码，未提交的改动在 checkout/rebase 后会消失
- [ ] **跑的 build 和数据库一致**：`select max(name) from kysely_migrations` 对照 `server/src/schema/migrations/ORDER`（上次 `package.json` 写 2.6.1，实际 build 来自更早的提交，有 8 个迁移从未执行）
- [ ] **unit 文件和仓库里的副本一致**（`systemctl cat immich-*` 对比 `systemd/`）

## 1. 选目标版本

- 用**正式 tag**，不用 `main`，也不用 RC：补丁版本从 release 分支打，`main` 上混着未发布的改动；迁移一旦跑了就回不去
- 通读两版之间所有 release notes（`gh release view <tag> -R immich-app/immich`），重点找：
  - Python / Node / pnpm 版本要求（v3.3 起 ML 需要 Python ≥ 3.12）
  - 目录结构变化（v3.2：SDK 移到 `packages/sdk`，插件变成 `packages/plugin-core`，Makefile 删除）
  - 需要手动重跑的任务（v3：Extract Metadata）
  - 手机 App 兼容性（v3 App 能连 v2 服务器，反过来不行 → **先升级手机**）
- ML 相关改动要单独看：`git log --oneline <old>..<new> -- machine-learning/immich_ml`。重写级别的变化（如 v3.3）需要专门移植，不要指望 rebase 自动解决

## 2. 在 worktree 里应用补丁

```bash
cd /home/radxa/immich-cix/immich-cix          # 主 checkout（不要在这里切分支：生产 ML 从磁盘读代码）
git fetch upstream --tags
git worktree add -b cix-vX.Y.Z /home/radxa/immich-cix/immich-vX vX.Y.Z
cd /home/radxa/immich-cix/immich-vX
for c in $(git log --reverse --format=%H vOLD..cix-vOLD -- . ':!cix'); do git cherry-pick $c || break; done
git checkout cix-vOLD -- cix                  # 文档目录直接带过来
```

先用 `git merge-tree --write-tree --name-only vX.Y.Z cix-vOLD` 预估冲突，不碰工作区。

已知冲突热点：

| 文件 | 处理 |
|---|---|
| `packages/sdk/src/fetch-client.ts` | 生成文件：取上游版本，加上 `V4L2M2M = "v4l2m2m"`，最后用 oazapfts 重新生成确认**无差异**（见第 4 步） |
| `machine-learning/immich_ml/models/base.py` | 上游常改 `_make_session` / 下载逻辑；按新结构重新加 `.cix` 分支 |
| `machine-learning/immich_ml/models/facial_recognition/recognition.py` | 上游 v3.2 去掉了 insightface；我们旧的 `_CixArcFaceWrapper` 已不需要，取上游版本 |

## 3. git 不会报的语义适配（逐项检查）

自动合并"干净"不代表能用。上次 `media.ts` 合并无冲突，但会让 HLS 实时转码全部失败。

**Server**
- [ ] `utils/media.ts`：`BaseConfig` / `BaseHWConfig` 的构造参数、可覆盖的方法名有没有变（v3.2 加了 `tune`、`getEncoderOptions()`，删了 `getSupportedCodecs()`）
- [ ] ffmpeg 参数必须**每个元素一个参数**（`['-b:v', '8000k']`，不能 `'-b:v 8000k'`）：HLS 路径直接 spawn ffmpeg，不经过 fluent-ffmpeg 的拆分
- [ ] 所有 `Record<TranscodeHardwareAcceleration, …>` 都有 `V4l2m2m` 条目（`pnpm check` 会报）
- [ ] 上游 `getToneMapping()` / `getScaling()` / `isFrameVertical()`（v3.3 新增）的语义变化是否影响 V4L2M2M 的覆盖（我们在硬解路径自己算尺寸）
- [ ] `CLIP_MODEL_INFO` 里仍有 `chinese-clip-*`
- [ ] Web：设置页文件可能搬家（v3.2：`web/src/routes/admin/system-settings/`）；CLIP 模型名是自由输入框，不需要改

**ML**
- [ ] session 的选择/注册方式（v3.2：`base.py` 里按后缀 `match`；v3.3：`ModelFormat` + session/graph 协议，见 [npu-ml.md](findings/npu-ml.md#v33-的-ml-重写)）
- [ ] 模型路径与默认格式：`model_path_for_format`、`_model_format_default`（CIX 优先）、`_download` 的 ignore patterns
- [ ] 各模型的预处理是否变了（人脸：v3.2 起 `_ops.py`；OCR：v3.2 起固定走 ORT）
- [ ] `OpenClipVisualEncoder.transform` 是否仍支持 `resize_mode: squash`
- [ ] `conftest.py` 的 `cix_unavailable` 自动 fixture 还在（否则在板子上跑上游测试会有十几个失败）

## 4. 构建

```bash
pnpm install --frozen-lockfile --filter immich --filter immich-web --filter @immich/sdk \
  --filter @immich/plugin-sdk --filter @immich/plugin-core --filter '@immich/i18n...'
pnpm --filter @immich/sdk --filter @immich/plugin-sdk build
pnpm --filter immich build && pnpm --filter immich-web build
PATH=/home/radxa/immich-cix/tools/binaryen-version_124/bin:$PATH pnpm --filter @immich/plugin-core build

# OpenAPI / SDK：重新生成后 git diff 应为空
(cd server && node ./dist/bin/sync-open-api.js)
npx -y oazapfts@7.5.0 --optimistic --argumentStyle=object --useEnumType --allSchemas \
  open-api/immich-openapi-specs.json packages/sdk/src/fetch-client.ts   # 不要再跑 prettier

# 新的 IMMICH_BUILD_DATA
B=/home/radxa/immich-build-vX && mkdir -p $B/plugins/immich-plugin-core/dist
cp -r web/build $B/www && cp -r /home/radxa/immich-build-v3/geodata $B/
cp packages/plugin-core/manifest.json $B/plugins/immich-plugin-core/
cp packages/plugin-core/dist/plugin.wasm $B/plugins/immich-plugin-core/dist/

# ML venv（Python 版本按上游 requires-python）
cd machine-learning && UV_PYTHON=<python> uv sync --frozen --extra cpu --group test
uv pip install --python .venv/bin/python /usr/share/cix/pypi/libnoe-3.0.0-py3-none-manylinux2014_aarch64.whl
```

工具链注意（详见 [deployment.md](findings/deployment.md#构建工具链)）：pnpm 会按 `packageManager` 自动切版本；extism-js 官方 1.7.0 二进制要 glibc 2.39，用本机 1.1.0 可以。

## 5. 测试

```bash
cd server && pnpm check && pnpm test && npx eslint <改过的文件> --max-warnings 0
cd ../machine-learning && LD_LIBRARY_PATH=/usr/share/cix/lib .venv/bin/python -m pytest test_main.py   # 不要裸跑 pytest
.venv/bin/python -m ruff check immich_ml test_main.py conftest.py && .venv/bin/python -m ruff format --check immich_ml
```

## 6. 在板子上验证硬件路径

工具都在 [tools/](tools/README.md)：

- [ ] `tools/ffmpeg/check_v4l2m2m.sh`：CIX ffmpeg 的坑是否还在（换 ffmpeg 包时必跑）
- [ ] 在 3013 端口起新版 ML（演练脚本会起），`tools/ml/compare_servers.py` 对比生产 3003：CLIP 应 cos=1.0、人脸框 IoU=1.0
- [ ] `tools/ml/embeddings_vs_db.py clip|faces`：新 embedding 和数据库里存的比（人脸用固定样本，才能跨版本对比）
- [ ] `tools/ml/ocr_check.py`：OCR 结果和数据库对照
- [ ] CixSession 并发：多线程同时 `run()` 结果要和串行一致（见 [npu-ml.md](findings/npu-ml.md#线程安全)）

## 7. 演练（数据库副本 + 只读媒体库）

```bash
WORKTREE=/home/radxa/immich-cix/immich-vX BUILD_DATA=/home/radxa/immich-build-vX \
MEDIA_LOCATION=<生产 IMMICH_MEDIA_LOCATION> DUMP=<pg_dump -Fc 文件> cix/tools/dry-run/start.sh
```

检查：
- [ ] `journalctl -u immich-dryrun`：迁移全部成功、mount checks 通过、插件加载、`tonemapx` 回退日志
- [ ] API 冒烟：统计、人物、缩略图、智能搜索、系统配置（API key 的创建方法见 [deployment.md](findings/deployment.md#用-api-自动化)）
- [ ] 对几段 HDR / 旋转视频跑 `refresh-metadata` → `regenerate-thumbnail` → `transcode-video`，用 ffprobe 检查输出（分辨率、`rotation`、bt709/tv）并抽帧看一眼
- [ ] 结束后 `cix/tools/dry-run/stop.sh`（不要让它过夜：夜间任务会扫整个 HDD）

## 8. 切换

1. 手机 App 先升级
2. **预检路径**：新 unit 里引用的每个路径都 `ls` 一遍（上次 `IMMICH_BUILD_DATA` 写错了目录，是预检发现的）
3. `systemctl stop immich-server immich-ml`
4. 最终备份：`podman exec immich_postgres pg_dump -U postgres -Fc immich > …/pre-vX-cutover-<ts>.pgdump` + `sudo zfs snapshot tank@immich-pre-vX-<ts>`
5. 换 unit 文件 → `daemon-reload` → **先起 ML 再起 server**（server 一起来就会恢复中断的任务，ML 没就绪会报 "Machine learning repository not been setup"）
6. 看迁移日志、外网访问（经 Caddy）、`/api/server/version`
7. 按 release notes 做升级后任务（v3：**Extract Metadata → All**），再按需跑缩略图/转码/OCR 的 Missing
8. 手动触发一次数据库备份，确认 `Database Backup Success`

## 9. 回滚

只在新版还没收到重要新数据前可行：

```bash
sudo systemctl stop immich-server immich-ml
sudo cp <旧 unit 文件> /etc/systemd/system/ && sudo systemctl daemon-reload
podman exec immich_postgres psql -U postgres -c "DROP DATABASE immich WITH (FORCE);" -c "CREATE DATABASE immich;"
podman exec -i immich_postgres pg_restore -U postgres -d immich < pre-vX-cutover-<ts>.pgdump
sudo systemctl start immich-ml immich-server
```

## 10. 收尾

- [ ] 在 `upgrade-logs/` 加一篇记录（时间线、遇到的坑、数据）
- [ ] 更新本目录里过时的内容和本机 `CLAUDE.md`
- [ ] 稳定 1–2 周后：删 ZFS 快照、旧 worktree/checkout、演练环境

## 以后用 CI 自动跟进（规划）

CI 能做：上游发新 tag 时自动 cherry-pick 补丁、在 GitHub ARM64 runner 上跑 `pnpm check/test` 和 ML 测试（NPU/VPU 已 mock）、成功开 PR / 冲突或失败开 issue。
CI 做不了：硬件验证（第 6、7 步必须在板子上）、部署（保持手动）、ML 架构级重写的移植（如 v3.3）。
不建议把板子注册成公开仓库的 self-hosted runner。
