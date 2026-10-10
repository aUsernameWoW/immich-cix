#!/bin/bash
# Start a sandboxed dry run of a new Immich build against a COPY of the production database.
#
#   - test PostgreSQL container on $TEST_DB_PORT, restored from $DUMP on first start
#   - server on $TEST_PORT as a transient systemd unit: /mnt/tank (the HDD pool) is read-only, the photo
#     library is an overlayfs (production files as the lower layer, writes go to $WORK/overlay/upper),
#     Redis DB 1 (production uses DB 0, so production jobs are never picked up)
#   - ML on $TEST_ML_PORT with its own model cache copy
#
# Usage: WORKTREE=... BUILD_DATA=... MEDIA_LOCATION=... DUMP=<file.pgdump> ./start.sh
set -euo pipefail

WORKTREE="${WORKTREE:?path of the worktree with the new build}"
BUILD_DATA="${BUILD_DATA:?IMMICH_BUILD_DATA of the new build (www/, plugins/, geodata/)}"
WORK="${WORK:-/home/radxa/immich-cix/dry-run}"
LIBRARY="${LIBRARY:-/mnt/tank/immich-library}"                       # real path behind the media symlink
MEDIA_LOCATION="${MEDIA_LOCATION:?IMMICH_MEDIA_LOCATION of production (paths stored in the DB point there)}"
PG_IMAGE="${PG_IMAGE:-ghcr.io/immich-app/postgres:14-vectorchord0.4.3-pgvectors0.2.0}"
NODE_BIN="${NODE_BIN:-/home/radxa/.nvm/versions/node/v24.18.0/bin}"
TEST_PORT="${TEST_PORT:-2284}" TEST_ML_PORT="${TEST_ML_PORT:-3013}" TEST_DB_PORT="${TEST_DB_PORT:-5433}"
DB_CONTAINER=immich_postgres_dryrun

mkdir -p "$WORK"/{pgdata,overlay/upper,overlay/work,overlay/merged,ml}

if ! podman container exists $DB_CONTAINER; then
  # host networking: podman's old pasta stalls large transfers (see findings/deployment.md);
  # with host networking postgres would otherwise listen on every interface, and this is a full copy of the DB
  podman run -d --name $DB_CONTAINER --network host \
    -e POSTGRES_PASSWORD=postgres -e POSTGRES_USER=postgres -e POSTGRES_DB=immich \
    -v "$WORK/pgdata:/var/lib/postgresql/data" --shm-size=128mb \
    "$PG_IMAGE" postgres -c config_file=/etc/postgresql/postgresql.conf -c port="$TEST_DB_PORT" -c listen_addresses=localhost
  until podman exec $DB_CONTAINER pg_isready -U postgres -p "$TEST_DB_PORT" -q; do sleep 1; done
  sleep 3
  podman exec -i $DB_CONTAINER pg_restore -U postgres -p "$TEST_DB_PORT" -d immich --exit-on-error < "${DUMP:?pg_dump -Fc of production}"
else
  podman start $DB_CONTAINER
fi

mountpoint -q "$WORK/overlay/merged" || sudo mount -t overlay overlay \
  -o "lowerdir=$LIBRARY,upperdir=$WORK/overlay/upper,workdir=$WORK/overlay/work" "$WORK/overlay/merged"

[ -d "$WORK/ml/cache" ] || cp -a "$HOME/.cache/immich_ml" "$WORK/ml/cache"
cd "$WORKTREE/machine-learning"
# immich_ml starts gunicorn as "python", so the venv has to come first in PATH
PATH="$PWD/.venv/bin:$PATH" MACHINE_LEARNING_CACHE_FOLDER="$WORK/ml/cache" IMMICH_HOST=127.0.0.1 \
  IMMICH_PORT="$TEST_ML_PORT" MACHINE_LEARNING_WORKERS=1 MACHINE_LEARNING_MODEL_TTL=0 LD_LIBRARY_PATH=/usr/share/cix/lib \
  nohup .venv/bin/python -m immich_ml > "$WORK/ml/ml.log" 2>&1 &

sudo systemd-run --unit=immich-dryrun --description="Immich dry run (test DB, overlay library)" \
  --uid="$(id -u)" --gid="$(id -g)" --working-directory="$WORKTREE/server" \
  -p ReadOnlyPaths=/mnt/tank -p ReadOnlyPaths=/data \
  -p BindPaths="$WORK/overlay/merged:$LIBRARY" -p ReadWritePaths="$LIBRARY" \
  -E HOME="$HOME" -E NODE_ENV=production -E PATH="$NODE_BIN:/usr/local/bin:/usr/bin:/bin" \
  -E DB_HOSTNAME=localhost -E DB_PORT="$TEST_DB_PORT" -E DB_USERNAME=postgres -E DB_PASSWORD=postgres -E DB_DATABASE_NAME=immich \
  -E REDIS_HOSTNAME=localhost -E REDIS_DBINDEX=1 -E IMMICH_MACHINE_LEARNING_URL="http://127.0.0.1:$TEST_ML_PORT" \
  -E IMMICH_MEDIA_LOCATION="$MEDIA_LOCATION" -E IMMICH_BUILD_DATA="$BUILD_DATA" -E IMMICH_PORT="$TEST_PORT" \
  -E IMMICH_HOST=127.0.0.1 \
  "$NODE_BIN/node" dist/main.js

echo "server: http://localhost:$TEST_PORT   logs: sudo journalctl -u immich-dryrun -f   ML log: $WORK/ml/ml.log"
