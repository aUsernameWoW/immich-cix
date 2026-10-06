#!/bin/bash
# Re-check the CIX ffmpeg quirks that the V4L2M2M code works around (see cix/findings/ffmpeg-vpu.md).
# Run it against every new CIX ffmpeg build before relying on it:
#   ./check_v4l2m2m.sh [/path/to/ffmpeg]            (e.g. /usr/share/cix/bin/ffmpeg with LD_LIBRARY_PATH=/usr/share/cix/lib)
set -uo pipefail
FF="${1:-ffmpeg}"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
SRC=(-f lavfi -i testsrc2=size=1920x1080:rate=30 -t 3)

echo "== $($FF -hide_banner -version | head -1)"
# list once: with pipefail, `grep -q` exiting early would SIGPIPE the producer and look like "not found"
FILTERS=$($FF -hide_banner -filters 2>/dev/null | awk '{print $2}')
for f in tonemapx zscale tonemap tonemap_opencl; do
  printf "filter %-15s %s\n" "$f" "$(grep -qx "$f" <<< "$FILTERS" && echo yes || echo NO)"
done
echo "encoders: $($FF -hide_banner -encoders 2>/dev/null | awk '/v4l2m2m/{print $2}' | paste -sd' ')"
echo "decoders: $($FF -hide_banner -decoders 2>/dev/null | awk '/v4l2m2m/{print $2}' | paste -sd' ')"

encode() { # name, extra args -> bitrate kbps and PSNR against the source
  local name=$1; shift
  if ! timeout 60 $FF -hide_banner -loglevel error -y "${SRC[@]}" -vf format=yuv420p "$@" "$T/$name.mp4"; then
    echo "  $name: FAILED"; return
  fi
  local br psnr
  br=$(ffprobe -v error -show_entries format=bit_rate -of csv=p=0 "$T/$name.mp4")
  psnr=$(ffmpeg -hide_banner -i "$T/$name.mp4" "${SRC[@]}" -lavfi "[1:v]format=yuv420p[b];[0:v][b]psnr" -f null - 2>&1 | grep -o 'average:[0-9.inf]*')
  printf "  %-28s %6d kbps  %s\n" "$name" $((br / 1000)) "$psnr"
}
echo "-- QP handling (expect: '-qp 23' == default ~1.2 Mbps unless '-rc_enable 0'; PSNR should be > 30 dB)"
encode h264-default -c:v h264_v4l2m2m
encode h264-qp23 -c:v h264_v4l2m2m -qp 23
encode h264-rc0-qp23 -c:v h264_v4l2m2m -rc_enable 0 -qp 23
encode h264-b4M -c:v h264_v4l2m2m -b:v 4M
encode hevc-rc0-qp28 -c:v hevc_v4l2m2m -rc_enable 0 -qp 28

echo "-- HW decode + scale (expect: '-2' fails because the size is forwarded to the decoder downscaler)"
timeout 60 $FF -hide_banner -loglevel error -y "${SRC[@]}" -vf format=yuv420p -c:v hevc_v4l2m2m -b:v 8M "$T/src.mp4"
for vf in "scale=1280:720" "scale=-2:720" "scale=w=-2:h=720"; do
  timeout 60 $FF -hide_banner -y -c:v hevc_v4l2m2m -noautorotate -i "$T/src.mp4" -c:v h264_v4l2m2m -rc_enable 0 -qp 28 \
    -vf "$vf,format=yuv420p" "$T/scaled.mp4" > "$T/log" 2>&1
  printf "  %-20s rc=%-3s %s\n" "$vf" "$?" "$(grep -m1 -oE "parameter 'dsl[wh]'[^.]*|Error[^:]*" "$T/log")"
done
