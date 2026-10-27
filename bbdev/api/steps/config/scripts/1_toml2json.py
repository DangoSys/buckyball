#!/usr/bin/env python3
"""Recursively expand one TOML entry and every include / *.toml reference into JSON."""

from __future__ import annotations

import json
import re
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


IP_DEFAULTS = {
    "uncached": "arch/src/main/scala/framework/core/memory/uncached/configs/default.toml",
    "fetch": "arch/src/main/scala/framework/core/execution/fetch/src/main/scala/configs/default.toml",
    "fpu": "arch/src/main/scala/framework/core/execution/fpu/src/main/scala/configs/default.toml",
    "controller": "arch/src/main/scala/framework/core/controller/configs/default.toml",
    "axi": "arch/src/main/scala/framework/ip/axi4/configs/default.toml",
    "requesters": "arch/src/main/scala/framework/ip/coherence/configs/requesters.toml",
    "ddr": "arch/src/main/scala/framework/device/ddr/configs/default.toml",
    "bootrom": "arch/src/main/scala/framework/device/peripherals/bootrom/configs/default.toml",
    "clint": "arch/src/main/scala/framework/device/peripherals/clint/configs/default.toml",
    "plic": "arch/src/main/scala/framework/device/peripherals/plic/configs/default.toml",
    "scu": "arch/src/main/scala/framework/device/peripherals/scu/configs/default.toml",
    "frontend": "arch/src/main/scala/framework/acc/frontend/configs/default.toml",
    "memdomain": "arch/src/main/scala/framework/acc/memdomain/configs/default.toml",
    "gpdomain": "arch/src/main/scala/framework/acc/gpdomain/configs/default.toml",
    "transport": (
        "arch/src/main/scala/framework/acc/"
        "memdomain/src/main/scala/tomem/configs/default.toml"
    ),
}
CONFIG_REFERENCES = set(IP_DEFAULTS) | {
    "device",
    "sims",
    "accelerator",
    "balldomain",
    "cpu",
}


def _reference(base: Path, value: str, slot: str | None) -> Path:
    if value == "default":
        if slot not in IP_DEFAULTS:
            _die(f"{base}: {slot!r} requires an explicit configuration path")
        return Path(__file__).resolve().parents[5] / IP_DEFAULTS[slot]
    return (base / value).resolve()


def _merge(left: dict, right: dict, path: Path) -> dict:
    result = dict(left)
    for key, value in right.items():
        if key == "_file":
            continue
        if key not in result:
            result[key] = value
        elif isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value, path / key)
        else:
            _die(f"{path}: include key collision: {key}")
    return result


def _walk_file(
    path: Path,
    includes: list[str],
    stack: list[str],
    override: bool = False,
) -> dict[str, Any]:
    path = path.resolve()
    key = str(path)
    if key in stack:
        _die("include cycle: %s" % " -> ".join(stack + [key]))
    if key not in includes:
        includes.append(key)
    data = _load(path)
    stack.append(key)
    out = _walk(data, path.parent, includes, stack, override=override)
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
    slot: str | None = None,
    override: bool = False,
) -> Any:
    if isinstance(obj, dict):
        inc = obj.get("include")
        if inc is not None:
            if not isinstance(inc, str) or not inc:
                _die(f"{base}: include must be a non-empty string")
            loaded = _walk_file(_reference(base, inc, slot), includes, stack)
            rest = {k: v for k, v in obj.items() if k != "include"}
            walked = _walk(rest, base, includes, stack, override=override)
            if not isinstance(walked, dict):
                _die(f"{base}: include siblings must be a table")
            if override:
                # Inside a device config, include siblings replace included keys.
                _override(loaded, walked, str(base / inc))
                return loaded
            return _merge(loaded, walked, base / inc)
        result = {}
        for key, value in obj.items():
            if key == "config" and value == "default" and "ip" in obj:
                repo = Path(__file__).resolve().parents[5]
                catalog = _load(repo / "arch/uvm.toml")
                path = repo / catalog["paths"][obj["ip"]] / "configs/uvm.toml"
                result[key] = _walk_file(path, includes, stack)
            elif key == "param" and value == "default":
                component = obj["component"]
                if not isinstance(component, str) or not component.isidentifier():
                    _die(f"{base}: invalid Ball component {component!r}")
                path = (
                    Path(__file__).resolve().parents[5]
                    / "examples/prefab/balls"
                    / component
                    / "configs/default.toml"
                )
                result[key] = _walk_file(path, includes, stack)
            else:
                # A device config holds overrides of the design config.
                result[key] = _walk(
                    value, base, includes, stack, key, override or key == "configs"
                )
        return result
    if isinstance(obj, list):
        return [_walk(x, base, includes, stack, slot, override) for x in obj]
    if isinstance(obj, str) and obj == "default" and slot in CONFIG_REFERENCES:
        return _walk_file(_reference(base, obj, slot), includes, stack)
    if isinstance(obj, str) and obj.endswith(".toml"):
        path = (base / obj).resolve()
        if not path.is_file():
            _die(f"toml path does not exist: {path} (from {base} + {obj!r})")
        return _walk_file(path, includes, stack, override)
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
        if not isinstance(raw.get("device"), dict):
            _die(f"{path}: missing [device]")
        raw["device"] = {"include": design.relative_to(path.parent).as_posix()}
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


def design2json(top: Path) -> dict[str, Any]:
    data = toml2json(top)
    data["workloads"] = _walk_file(
        Path(top).resolve().parent.parent / "workloads/configs/tests.toml",
        data["includes"],
        [],
    )
    return data


def _override(config: dict, override: dict, path: str) -> None:
    for key, value in override.items():
        if key == "_file":
            continue
        if key not in config:
            _die(f"{path}: override key {key!r} is not in the design config")
        if isinstance(config[key], dict) and isinstance(value, dict):
            _override(config[key], value, f"{path}.{key}")
        else:
            config[key] = value


def simulations(data: dict[str, Any]) -> list[tuple[str, str | None, str]]:
    """(entry, config, selection) of every simulation of the design: one per
    config registered on the device entry, one per core entry."""
    design = Path(data["_file"]).parent.parent.name
    prefix = "".join(part.capitalize() for part in re.split("[-_]", design))
    result = []
    for entry, registration in data["sims"]["targets"].items():
        if entry == "_file":
            continue
        if entry != "device":
            if "configs" in registration:
                _die(f"targets.{entry}: only the device entry registers configs")
            result.append((entry, None, entry))
            continue
        if not registration["configs"]:
            _die("targets.device.configs registers no config")
        for name in registration["configs"]:
            if not re.fullmatch(prefix + "[A-Za-z0-9]*Config", name):
                _die(f"targets.device.configs: {name!r} must be {prefix}<Name>Config")
            result.append((entry, name, f"device/{name}"))
    return result


def simulation2json(
    data: dict[str, Any], entry: str, name: str | None, selection: str
) -> dict[str, Any]:
    """The config of one simulation: the design config with the tables of device
    config `name` replacing the same keys (arrays whole)."""
    config = json.loads(json.dumps(data))
    registration = config["sims"]["targets"][entry]
    if name is not None:
        overrides = registration.pop("configs")[name]
        _override(config, overrides, overrides["_file"])
    config["sims"]["targets"] = {entry: registration}
    config["sims"]["selection"] = selection
    return config


def main() -> None:
    if len(sys.argv) != 3:
        _die(f"usage: {sys.argv[0]} DESIGN.toml SELECTION")
    data = design2json(Path(sys.argv[1]))
    found = [s for s in simulations(data) if s[2] == sys.argv[2]]
    if not found:
        _die(f"{sys.argv[1]}: no simulation {sys.argv[2]!r}")
    out = simulation2json(data, *found[0])
    sys.stdout.write(json.dumps(out, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
