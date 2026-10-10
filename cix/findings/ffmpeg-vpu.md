# CIX ffmpeg / VPU 的坑

验证日期 2026-10-06，板子：Orion O6，内核 6.1.44，`ffmpeg 8:5.1.6-0+deb12u1+cix`。
所有结论都可以用 [`tools/ffmpeg/check_v4l2m2m.sh`](../tools/ffmpeg/check_v4l2m2m.sh) 重新验证。

## 必须用 CIX 的 ffmpeg

| ffmpeg | `h264_v4l2m2m` | `hevc_v4l2m2m` | `tonemapx` |
|---|---|---|---|
| `/usr/bin/ffmpeg`（CIX 5.1.6） | ✓ PSNR 40.8 dB | ✓ PSNR 41.2 dB | ✗ |
| jellyfin-ffmpeg 8.1.2（`/usr/lib/jellyfin-ffmpeg/`，已安装） | 能跑但**输出是花屏**（PSNR ≈ 10 dB） | ✗ 只写出 44 字节 | ✓ |
| CIX 26Q2 `cix-ffmpeg 5.1.7`（装到 `/usr/share/cix/{bin,lib}`，与系统版并存） | 与 5.1.6 行为完全一致 | 同左 | ✗ |

结论：上游 FFmpeg 的通用 V4L2M2M 实现驱动不了 `mvx`（Linlon）VPU，只能用 CIX 打过补丁的版本；26Q2 的新包没有带来任何行为变化，不急着换。

设备：编码器 `/dev/video4`，解码器 `/dev/video-cixdec0`（ffmpeg 自动探测）。V4L2M2M 不需要 `/dev/dri`，所以 `V4l2m2mSwDecodeConfig.getDevice()` 返回空串，避免 `BaseHWConfig` 在没有 DRI 设备时抛错。

## 1. `-qp` 在码率控制开着时被静默忽略

编码器有 `rc_enable`（默认 1）。只传 `-qp 23` 时输出和什么都不传**完全一样**（约 1.2 Mbps）。
旧代码因此一直以约 1.2 Mbps 转码，CRF 设置从未生效。

| 参数（testsrc2 1080p30） | 码率 | PSNR |
|---|---|---|
| 默认 / `-qp 23` | 1265 kbps | 31.7 dB |
| `-rc_enable 0 -qp 23` | 11184 kbps | 43.7 dB |
| `-b:v 6M` | 6163 kbps | 40.8 dB |

修复：CQP 模式输出 `-rc_enable 0 -qp <crf>`。`-qpi/-qpp/-qpb` 不需要，`-qp` 已覆盖 I/P/B。

### 硬件 QP ≠ x264 CRF

真实素材（4K PQ 手机视频 → 720p60，与 CPU 色调映射后的参考帧比较）：

| 设置 | 码率 | PSNR |
|---|---|---|
| V4L2M2M QP 23 | 21.8 Mbps | 40.1 dB |
| V4L2M2M QP 28 | 9.7 Mbps | 36.6 dB |
| V4L2M2M `-b:v 4M` | 4.05 Mbps | 34.1 dB |
| libx264 CRF 23（参考） | 5.9 Mbps | 36.1 dB |

所以"CRF 值直接当 QP"（上游 RKMPP 也这么做）会比软件编码大 3–4 倍。生产设为 **CRF 28**。

## 2. 硬解时 `scale` 尺寸被转发给解码器的硬件缩放（`dslw`/`dslh`）

CIX 的 `*_v4l2m2m` 解码器有 `dslw`/`dslh`（硬件缩小）选项，CIX 的 fftools 会把 `-vf` 里 `scale` 的尺寸自动塞进去：

| `-vf` | 结果 |
|---|---|
| `scale=1280:720` | ✓（缩放在解码器里完成） |
| `scale=-2:720` | ✗ `Value -2.000000 for parameter 'dslw' out of range` |
| `scale=w=-2:h=720` | ✗ **段错误** |

Immich 生成的是 `-2:720`，于是旧代码的**硬解 + 缩放全部失败**，靠 Immich 的回退链（软解 → 纯软件）才转出来；HDR 视频则每一级都失败。
生产升级前：4119 个视频里只有 144 个有转码版本。

修复（`V4l2m2mHwDecodeConfig.getScaling`）：用了硬件解码器时，自己算出偶数的显式宽高。

## 3. 旋转：硬解保留 `-noautorotate`，按存储方向算尺寸

- 硬件解码器输出的帧没有旋转；CIX 5.1 在 `-noautorotate` 下会把 display matrix 原样带到输出（实测 -180°、90° 都保留），播放器负责旋转
- 硬件缩放作用于未旋转的帧，所以尺寸必须按**存储方向**算（`getOutputSize({...videoStream, rotation: 0})`），不能用考虑旋转的 `isVideoVertical`
- 软解路径（包括硬解不支持的编码）**不加** `-noautorotate`，让 ffmpeg 自己转正，再用上游的 `getScaling`
- 上游 v3.2.4 的 RKMPP 硬解也有"旋转视频按旋转后方向缩放"的 bug，v3.3 用 `isFrameVertical()` 修了；跟进 v3.3 时注意它和我们覆盖的关系

## 4. 没有 `tonemapx`

`tonemapx` 是 jellyfin-ffmpeg 独有的滤镜。上游的 HDR 转码和**视频缩略图**都用它，所以在 CIX ffmpeg 上 HDR 视频的缩略图每晚失败（生产上约 150 个/晚，`No such filter: 'tonemapx'` → `Error reinitializing filters`）。

修复：启动时检测（`MediaRepository.hasFilter('tonemapx')`），没有就退回：

```
zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=<algo>:desat=0,zscale=t=bt709:m=bt709:r=<range>,format=yuv420p
```

- 缩略图 / 软件转码用 `r=pc`（与上游 `tonemapx ... r=pc` 一致）
- V4L2M2M 用 `r=tv`：编码器不在码流里标注 full range
- 速度（4K HLG → 720p）：硬解 + CPU zscale 约 15 fps；软解约 6–8 fps。瓶颈是 float 色调映射
- HEVC 10-bit 硬解输出 P010（不会先截成 8 bit），适合色调映射

## 5. OpenCL 色调映射不可用

- CIX ffmpeg 有 `tonemap_opencl`，但没有 `tonemap_mode` 选项（上游 RKMPP 代码用的是 jellyfin 的扩展）
- 系统里的 OpenCL ICD 只有 Mesa / rusticl，没有 `mali.icd`，Immich 检测到 `mali=false`
- 所以 V4L2M2M 的 OpenCL 分支已删除，统一 CPU zscale

## 6. 其他

- **`-level`**：编码器只接受命名档位（`-level 5.1`），传数字会报 `Error setting option level`。不传即可，编码器默认值合理
- **AV1 10-bit 硬解**会进入 POLLERR 死循环 → `pixelFormat !== 'yuv420p'` 的 AV1 走软解
- 编码器默认 `-bf 3`、`-g 120`；Immich 会传 `-g 256`（HW 默认）
- **视频缩略图 vs 转码**：缩略图走 `ThumbnailConfig`（总是软解 + 软件滤镜），和硬件加速设置无关
