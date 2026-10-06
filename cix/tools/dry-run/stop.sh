#!/bin/bash
# Stop the dry run started by start.sh (keeps its data; delete $WORK and the container to reset).
WORK="${WORK:-/home/radxa/immich-cix/dry-run}"
TEST_ML_PORT="${TEST_ML_PORT:-3013}"
sudo systemctl stop immich-dryrun 2>/dev/null
# gunicorn outlives the immich_ml launcher; match on the test port so production ML is never hit.
# The [-] keeps pkill from matching the shell running this line.
pkill -f "[-]b 127.0.0.1:$TEST_ML_PORT"
podman stop immich_postgres_dryrun >/dev/null 2>&1
mountpoint -q "$WORK/overlay/merged" && sudo umount "$WORK/overlay/merged"
echo stopped
