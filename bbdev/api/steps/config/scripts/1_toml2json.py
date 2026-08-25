#!/usr/bin/env python3
"""Recursively expand one TOML entry and every include / *.toml reference into JSON."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from typing import Any


def _die(msg: str) -> None:
    print(f"1_toml2json: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _die(f"missing TOML: {path}")
    with path.open("rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        _die(f"TOML root must be a table: {path}")
    return data


def _walk_file(
    path: Path,
    includes: list[str],
    stack: list[str],
) -> dict[str, Any]:
    path = path.resolve()
    key = str(path)
    if key in stack:
        _die("include cycle: %s" % " -> ".join(stack + [key]))
    if key not in includes:
        includes.append(key)
    data = _load(path)
    stack.append(key)
    out = _walk(data, path.parent, includes, stack)
    stack.pop()
    if not isinstance(out, dict):
        _die(f"{path}: root must be a table")
    if "_file" in out:
        _die(f"{path}: reserved key '_file'")
    out["_file"] = key
    return out


def _walk(
    obj: Any,
    base: Path,
    includes: list[str],
    stack: list[str],
) -> Any:
    if isinstance(obj, dict):
        inc = obj.get("include")
        if inc is not None:
            if not isinstance(inc, str) or not inc:
                _die(f"{base}: include must be a non-empty string")
            loaded = _walk_file((base / inc).resolve(), includes, stack)
            rest = {k: v for k, v in obj.items() if k != "include"}
            walked = _walk(rest, base, includes, stack)
            if not isinstance(walked, dict):
                _die(f"{base}: include siblings must be a table")
            overlap = (set(loaded) & set(walked)) - {"_file"}
            if overlap:
                _die(f"{base / inc}: include key collision: {sorted(overlap)}")
            merged = dict(loaded)
            merged.update(walked)
            return merged
        return {k: _walk(v, base, includes, stack) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(x, base, includes, stack) for x in obj]
    if isinstance(obj, str) and obj.endswith(".toml"):
        path = (base / obj).resolve()
        if not path.is_file():
            _die(f"toml path does not exist: {path} (from {base} + {obj!r})")
        return _walk_file(path, includes, stack)
    return obj


def toml2json(top: Path, design: Path | None = None) -> dict[str, Any]:
    path = Path(top)
    if not path.is_file():
        raise FileNotFoundError(f"missing TOML: {path}")
    path = path.resolve()
    includes: list[str] = []
    if design is None:
        data = _walk_file(path, includes, [])
    else:
        design = Path(design)
        if not design.is_file():
            raise FileNotFoundError(f"missing TOML: {design}")
        design = design.resolve()
        raw = _load(path)
        if not isinstance(raw.get("designs"), dict):
            _die(f"{path}: missing [designs]")
        raw["designs"] = {"include": design.relative_to(path.parent).as_posix()}
        includes.append(str(path))
        data = _walk(raw, path.parent, includes, [str(path)])
        if not isinstance(data, dict):
            _die(f"{path}: root must be a table")
        if "_file" in data:
            _die(f"{path}: reserved key '_file'")
        data["_file"] = str(path)
    if "includes" in data:
        _die(f"{path}: reserved key 'includes'")
    data["includes"] = includes
    return data


def main() -> None:
    if len(sys.argv) != 2:
        _die(f"usage: {sys.argv[0]} TOP.toml")
    out = toml2json(Path(sys.argv[1]))
    sys.stdout.write(json.dumps(out, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
