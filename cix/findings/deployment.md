# 裸机部署的坑

部署形态：systemd（`immich.target` → postgres / redis / ml / server），PostgreSQL 在 rootless podman 里，Node 用 nvm，ML 用独立 venv，server 同时 serve web 静态文件（`IMMICH_BUILD_DATA/www`）。
server 和 ML 都用 `IMMICH_HOST=127.0.0.1` 只监听本机：server 前面是反向代理，ML 只有 server 访问。ML 只监听 IPv4 回环，所以 server 的 `IMMICH_MACHINE_LEARNING_URL` 写 `127.0.0.1` 而不是 `localhost`。
上游的一切默认都假设 Docker 镜像，裸机上要自己补齐镜像里"自带"的东西。

## 数据库备份

**现象**：自动备份从 2025-08-05 起再没成功过，每晚日志 `spawn /usr/lib/postgresql/14/bin/pg_dump ENOENT`。
**原因与修复**：

- Immich 硬编码 `/usr/lib/postgresql/<服务端大版本>/bin/{pg_dump,psql}`，在**宿主机**上执行，环境变量只给 `PATH` 和 `PGPASSWORD`（没有 `HOME`/`XDG_RUNTIME_DIR`），所以写个 `podman exec` 包装脚本也不可靠
- Debian 12 只有 `postgresql-client-15` → 加 PGDG 源装 `postgresql-client-14`（只多装两个包，不动 libpq）：
  ```bash
  sudo install -d /usr/share/postgresql-common/pgdg
  sudo curl -fo /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc https://www.postgresql.org/media/keys/ACCC4CF8.asc
  echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt bookworm-pgdg main" \
    | sudo tee /etc/apt/sources.list.d/pgdg.list
  sudo apt-get update && sudo apt-get install postgresql-client-14
  ```
  国内的 TUNA/USTC 镜像没有 bookworm-pgdg，官方源可直连
- 每次失败的备份都会**泄漏一个 `gzip --rsyncable` 进程**（先 spawn gzip、pg_dump 再失败），累计了约 40 个；重启 server 后清掉
- 验证：用 Immich 完全相同的参数和环境跑一次，或者用 API 触发 `backupDatabase` 任务，看 `Database Backup Success`

## podman pasta 在背压下卡死

**现象**：装好 pg_dump 后，`pg_dump | gzip` 三次里卡死两次；服务端显示 `idle in transaction / ClientRead`，客户端在 poll。去掉 gzip（直接写 `/dev/null`）则 4 秒完成。
**原因**：Debian 12 的 `passt 0.0~git20230309`（rootless podman 的 pasta 网络）在 TCP 零窗口/背压后恢复不了。
**修复**：把 `immich_postgres` 容器改成 `--network host`（数据在 bind mount 里，重建容器不影响数据；停机不到 1 分钟）。之后 5/5 通过。其他 rootless 容器仍走 pasta，有大流量时可能同样受影响。

**注意监听地址**：`-p 5432:5432` 和 host 网络默认都监听所有网卡（含 IPv6）。家宽常有公网 IPv6，而 PostgreSQL 超级用户 + 弱口令等于把整台机器交出去（`COPY … FROM PROGRAM`）。
Immich 只从本机连数据库，所以应当加 `-c listen_addresses=localhost`（并换掉默认密码），或用防火墙只放行回环。演练脚本里的副本库同样只监听 localhost。

## 生产状态和"以为的"不一致

- `package.json` 说 2.6.1，但跑的 `dist/` 是更早构建的，**8 个 v2.6 迁移从未执行**。判断方法：`kysely_migrations` 对照 `server/src/schema/migrations/ORDER`
- ML 服务跑的是 worktree 里**未提交**的代码（包括适配 libnoe 3.0 的改动）；HEAD 的代码在当前 libnoe 上根本跑不起来
- 仓库里的 unit 副本和 `/etc/systemd/system/` 不一致；`install.sh` 一跑就会把 ML 改回旧的独立服务
- 标记为废弃的 `immich-web`（vite dev server，0.0.0.0:3000）仍被 `immich.target` 拉起
- ML 任务从 7 月中起就没成功过（新上传没有 CLIP/人脸/OCR），而生产 Python 环境里根本没装 `rapidocr`
- 文档里的 ML 部署方式、人脸模型名也都是错的

教训：每次动手前先核实，见 [REPATCH.md 第 0 步](../REPATCH.md#0-开工前先确认生产环境的真实状态)。

## 构建工具链（v3.2）

- **pnpm 11**：根 `package.json` 的 `packageManager` 会让本机 pnpm 10 自动切到 11.x；`pnpm-workspace.yaml` 用 `allowBuilds`
- **Node**：上游 pin 24.15，本机 24.18 可用；systemd 里写死 nvm 路径，升级 Node 要同步改 unit
- **Makefile 已删除**，上游用 mise 任务（本机没装 mise，直接用 pnpm 命令即可，见 [REPATCH.md 第 4 步](../REPATCH.md#4-构建)）
- **SDK** 在 `packages/sdk`；重新生成用 oazapfts 7.5.0，生成后**不要再跑 prettier**（会产生上万行无关 diff）
- **核心插件** `packages/plugin-core`：需要 `extism-js` 和 binaryen
  - 官方 extism-js 1.7.0 预编译二进制要 glibc 2.39（Debian 12 是 2.36），用不了；本机 1.1.0 构建出的 wasm 在 v3.2.4 上正常加载
  - binaryen version_124 放在 `/home/radxa/immich-cix/tools/`，构建时放进 PATH
- **`IMMICH_BUILD_DATA` 布局**：`www/`、`geodata/`、`plugins/immich-plugin-core/{manifest.json,dist/plugin.wasm}`（v2 是 `corePlugin/`）；v3.3 还要 `geodata/countryInfo.txt`（GeoNames：`https://download.geonames.org/export/dump/countryInfo.txt`；每次反向地理编码都会读它，缺了整个功能报错）
- 我自己在切换前犯的错：unit 里写的 `IMMICH_BUILD_DATA` 路径和实际目录不同，是切换前的预检发现的 → **切换前把 unit 里每个路径都 `ls` 一遍**
- `IMMICH_BUILD_DATA` 里没有 `build-lock.json`（Docker 镜像才有）：打开管理页的版本信息时 server 会 `WARN Failed to read …/build-lock.json`，然后退回执行 `ffmpeg -version` 等命令取版本，无害

## 迁移与升级后任务

- v2.6 → v3.2.4 共 40 个迁移，30k 资产、422 MB 数据库上 **12 秒**跑完；其中多个不可逆（`DropAuditTable`、`EncodedVideoAssetFiles`、`ChangeDurationToInteger` 等），从 v3.2 起服务器拒绝降级启动
- **Extract Metadata → All 是必做的**：v3 把视频流信息存进新表（`asset_video`/`asset_audio`/`asset_keyframe`），老视频在重提取前，缩略图和转码都会报 `Missing video metadata`。30k 资产用了 37 分钟
- 系统配置只存"与默认值不同"的字段：v3.2 起 `accelDecode` 默认 true，所以升级后它从存储里消失是正常的
- 换 CLIP 模型：服务器会清空 embedding（维度不同时还会改列类型），但**不会自动重建索引**，要手动跑 Smart Search → All
- 新的夜间完整性检查默认会对全库做 checksum，在 HDD 上负载很重，可以在设置里调
- v3.2.4 → v3.3.1：4 个迁移 2 秒跑完；没有必做的升级后任务。上游改进了缩略图的缩放算法，但只影响新生成的缩略图，想让老照片也受益要手动跑一次全部缩略图（可选）
- `regenerate-thumbnail` 会原地覆盖同一路径的文件，`asset_file.updatedAt` **不变**：判断有没有跑完要看文件的 mtime
- ML 日志里每隔一段时间会出现 `Shutting down due to inactivity` + `Worker … was sent SIGINT!`：这是上游的空闲回收（模型卸载后重启 worker 释放内存），不是崩溃

## 手机 App

v3 App 能连 v2 服务器，v2 App 连不上 v3 服务器（主版本不匹配，且删除了旧同步接口）→ **先升级所有手机再切服务器**。

## 已知未解决

- **216 张 DNG 没有缩略图**：Sony Xperia 1 V（XQ-BE72）拍的 DNG。sharp 自带的 libvips 没有 RAW 解码器，把 DNG 当 TIFF 读到原始 CFA 数据，报 `vips_colourspace: no known route from 'multiband' to 'rgb16'`；这些 DNG 也没有内嵌 JPEG 预览，所以 `extractEmbedded` 没用。修复需要自己编译带 libraw 的 libvips（≥8.16）并让 sharp 用它

## 演练环境的做法

[`tools/dry-run/start.sh`](../tools/dry-run/start.sh) 的隔离手段，可单独借用：

- 数据库：生产 dump 恢复到另一个容器（另一个端口）
- 媒体库：overlayfs（生产目录做只读下层，写入落到 upper），并用 `systemd-run -p ReadOnlyPaths=/mnt/tank -p BindPaths=<overlay>:<真实路径>` 让 server 看到的路径和数据库里存的一致
- Redis：`REDIS_DBINDEX=1`，不会抢生产（DB 0）的任务
- ML：独立端口 + 模型缓存副本（新版遇到缺失模型会下载或原地改写 ONNX）

## 用 API 自动化

需要管理员 API key 时，可以直接写进数据库（`key` 是 sha256 原始字节，v2.6 起为 `bytea`），用完**一定要删**：

```bash
KEY=$(openssl rand -hex 24); HASH=$(echo -n "$KEY" | sha256sum | cut -d' ' -f1)
podman exec immich_postgres psql -U postgres -d immich -c \
  "insert into api_key (name, key, \"userId\", permissions) values ('temp (delete me)', decode('$HASH','hex'), '<admin user id>', '{all}') returning id;"
curl -H "x-api-key: $KEY" localhost:2283/api/users/me
# 任务：PUT /api/jobs/<queue> {"command":"start","force":true|false}   队列状态：GET /api/queues/<queue>
# 单个资产：POST /api/assets/jobs {"assetIds":[...],"name":"refresh-metadata|regenerate-thumbnail|transcode-video|refresh-faces"}
# 删除：DELETE /api/api-keys/<id>
```

改系统配置要 GET 完整配置、只改一个字段、再整体 PUT，然后对比前后有效配置确认只改了这一处。

## Shell 小坑

- `pkill -f <pattern>` 会匹配到执行它的 shell 自己（命令行里含 pattern）→ 用 `'[-]b 127.0.0.1:3013'` 这种写法
- `set -o pipefail` + `grep -q`：grep 提前退出导致上游 SIGPIPE，管道被判失败 → 先把输出存进变量再 grep
- zsh 里单独的 `=====` 会被当成 `=` 展开报错
- zsh 里 `$q:wait` 的 `:w` 会被当成变量修饰符，拼出来的 key 是错的（看起来像队列为空）→ 写成 `${q}:wait`
- BullMQ 的队列在 Redis 里是 `immich_bull:<queue>:wait`（list）/ `:active`；数量以 `GET /api/queues/<queue>` 为准
- 数据库时间是 UTC；在 SQL 里写本地时间要带时区（`'2026-10-06 15:28+08'`）
