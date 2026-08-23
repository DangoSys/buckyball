#!/usr/bin/env python3
"""Flatten a TOML file and every include / *.toml path it references into one JSON document.

Usage:
  toml2json.py TOML OUT.json [--repo ROOT]
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any


def _die(msg: str) -> None:
    print(f"toml2json: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _die(f"missing TOML: {path}")
    with path.open("rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        _die(f"TOML root must be a table: {path}")
    return data


def _rel(path: Path, repo: Path | None) -> str:
    path = path.resolve()
    if repo is None:
        return str(path)
    return path.relative_to(repo.resolve()).as_posix()


def _walk_file(
    path: Path,
    includes: list[str],
    stack: list[str],
    repo: Path | None,
) -> dict[str, Any]:
    path = path.resolve()
    key = str(path)
    if key in stack:
        _die("include cycle: %s" % " -> ".join(stack + [key]))
    if not path.is_file():
        _die(f"missing TOML: {path}")
    rel = _rel(path, repo)
    if rel not in includes:
        includes.append(rel)
    data = _load(path)
    stack.append(key)
    out = _walk(data, path.parent, includes, stack, repo)
    stack.pop()
    if not isinstance(out, dict):
        _die(f"{path}: root must be a table")
    if "_file" in out:
        _die(f"{path}: reserved key '_file'")
    out["_file"] = rel
    return out


def _walk(
    obj: Any,
    base: Path,
    includes: list[str],
    stack: list[str],
    repo: Path | None,
) -> Any:
    if isinstance(obj, dict):
        inc = obj.get("include")
        if inc is not None:
            if not isinstance(inc, str) or not inc:
                _die(f"{base}: include must be a non-empty string")
            loaded = _walk_file((base / inc).resolve(), includes, stack, repo)
            rest = {k: v for k, v in obj.items() if k != "include"}
            walked = _walk(rest, base, includes, stack, repo)
            if not isinstance(walked, dict):
                _die(f"{base}: include siblings must be a table")
            overlap = (set(loaded) & set(walked)) - {"_file"}
            if overlap:
                _die(f"{base / inc}: include key collision: {sorted(overlap)}")
            merged = dict(loaded)
            merged.update(walked)
            return merged
        return {k: _walk(v, base, includes, stack, repo) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(x, base, includes, stack, repo) for x in obj]
    if isinstance(obj, str) and obj.endswith(".toml"):
        path = (base / obj).resolve()
        if not path.is_file():
            _die(f"toml path does not exist: {path} (from {base} + {obj!r})")
        return _walk_file(path, includes, stack, repo)
    return obj


def toml2json(path: Path, repo: Path | None = None) -> dict[str, Any]:
    includes: list[str] = []
    data = _walk_file(path, includes, [], repo)
    if "includes" in data:
        _die(f"{path}: reserved key 'includes'")
    data["includes"] = includes
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("toml", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--repo", type=Path)
    args = parser.parse_args()

    repo = args.repo.resolve() if args.repo is not None else None
    if repo is not None and not repo.is_dir():
        _die(f"repo is not a directory: {repo}")
    src = args.toml
    if not src.is_absolute():
        src = ((repo / src) if repo is not None else src).resolve()

    data = toml2json(src, repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
