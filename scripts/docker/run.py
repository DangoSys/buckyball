#!/usr/bin/env python3
"""Run a Buckyball Docker sandbox container.

Usage:
    python scripts/docker/run.py [--repo PATH] [--port PORT] [--name NAME] [--image IMAGE]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_IMAGE = "buckyball:latest"
DEFAULT_PORT = 3000


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Buckyball Docker sandbox container")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--name", default=None)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    args = parser.parse_args()

    repo = args.repo.resolve()
    name = args.name or f"bb-{repo.name}"

    print(f"[run] repo={repo}  port={args.port}  name={name}  image={args.image}")

    result = subprocess.run(
        ["docker", "run", "--rm", "-d",
         "-v", f"{repo}:/workspace",
         "-p", f"{args.port}:3000",
         "--name", name,
         args.image],
        check=True, capture_output=True, text=True,
    )
    container_id = result.stdout.strip()
    print(f"[run] container: {container_id[:12]}")

    url = f"http://localhost:{args.port}/health"
    print(f"[run] waiting for {url} ...")
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    break
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    else:
        print(f"[run] ERROR: server not ready after 120s", file=sys.stderr)
        print(f"[run] logs: docker logs {name}", file=sys.stderr)
        sys.exit(1)

    print(f"[run] ready: http://localhost:{args.port}")
    print(f"[run] shell: docker exec -it {name} bash")
    print(f"[run] stop:  docker stop {name}")


if __name__ == "__main__":
    main()
