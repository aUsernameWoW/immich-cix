"""Tiny helper to run read-only queries against the Immich database container."""

import os
import subprocess


def psql(query: str, sep: str = "|") -> list[list[str]]:
    container = os.environ.get("PG_CONTAINER", "immich_postgres")
    cmd = ["podman", "exec", container, "psql", "-U", os.environ.get("PG_USER", "postgres")]
    cmd += [
        "-p",
        os.environ.get("PG_PORT", "5432"),
        "-d",
        os.environ.get("PG_DB", "immich"),
        "-At",
        "-F",
        sep,
        "-c",
        query,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [line.split(sep) for line in out.strip().splitlines() if line]
