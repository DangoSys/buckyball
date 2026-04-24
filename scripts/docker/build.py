#!/usr/bin/env python3
"""Build the Buckyball Docker sandbox image.

Usage:
    python scripts/docker/build.py [--tag TAG]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Buckyball Docker image via nix")
    parser.add_argument("--tag", default="buckyball:latest")
    args = parser.parse_args()

    print(f"[build] nix build .#dockerImage")
    subprocess.run(
        ["nix", "build", ".#dockerImage", "--out-link", "result-docker"],
        cwd=REPO_ROOT,
        check=True,
    )

    image_path = (REPO_ROOT / "result-docker").resolve()
    print(f"[build] docker load < {image_path}")
    with open(image_path, "rb") as f:
        result = subprocess.run(["docker", "load"], stdin=f, check=True, capture_output=True)

    output = result.stdout.decode()
    print(output.strip())

    # Retag if the baked name differs from requested tag
    for line in output.splitlines():
        if line.startswith("Loaded image"):
            loaded = line.split(":", 1)[-1].strip()
            if loaded != args.tag:
                print(f"[build] docker tag {loaded} → {args.tag}")
                subprocess.run(["docker", "tag", loaded, args.tag], check=True)
            break

    print(f"[build] done: {args.tag}")


if __name__ == "__main__":
    main()
