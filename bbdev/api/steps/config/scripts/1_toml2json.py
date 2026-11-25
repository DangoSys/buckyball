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
    raise ValueError(f"1_toml2json: {msg}")


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _die(f"missing TOML: {path}")
    with path.open("rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        _die(f"TOML root must be a table: {path}")
    return data


IP_DEFAULTS = {
    "uncached": "arch/src/main/scala/hart/memory/uncached/configs/default.toml",
    "fetch": "arch/src/main/scala/mlp/fetch/src/main/scala/configs/default.toml",
    "fpu": "arch/src/main/scala/hart/execution/fpu/src/main/scala/configs/default.toml",
    "controller": "arch/src/main/scala/tlp/controller/configs/default.toml",
    "axi": "arch/src/main/scala/generic/axi4/configs/default.toml",
    "requesters": "arch/src/main/scala/generic/coherence/configs/requesters.toml",
    "ddr": "arch/src/main/scala/generic/ddr/configs/default.toml",
    "bootrom": "arch/src/main/scala/device/peripherals/bootrom/configs/default.toml",
    "clint": "arch/src/main/scala/device/peripherals/clint/configs/default.toml",
    "plic": "arch/src/main/scala/device/peripherals/plic/configs/default.toml",
    "scu": "arch/src/main/scala/device/peripherals/scu/configs/default.toml",
    "trace": "arch/src/main/scala/device/peripherals/trace/configs/default.toml",
    # The accelerator IPs: one copy in the hart (its specialization, custom/),
    # one in the thread (thread/acc/); `{acc}` is the owner's.
    "frontend": "arch/src/main/scala/{acc}/frontend/configs/default.toml",
    "memdomain": "arch/src/main/scala/{acc}/memdomain/configs/default.toml",
    "gpdomain": "arch/src/main/scala/{acc}/gpdomain/configs/default.toml",
    "transport": "arch/src/main/scala/{acc}/memdomain/tomem/configs/default.toml",
    # The host of a Ball alone (--ball).
    "host": "arch/src/main/scala/ball/configs/default.toml",
    # The host of a thread alone (--thread).
    "threadhost": "arch/src/main/scala/thread/configs/default.toml",
    # The host of a hart alone (--hart without --config).
    "harthost": "arch/src/main/scala/hart/configs/default.toml",
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
        # A thread's configuration lives under a `threads` directory, a hart's under `harts`.
        acc = "thread/acc" if "threads" in base.parts else "custom"
        return Path(__file__).resolve().parents[5] / IP_DEFAULTS[slot].format(acc=acc)
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
    # A file built on an include keeps the included file's `_file`: the
    # component config at the bottom of the chain locates the component.
    if "include" in data:
        return out
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
                # Inside a device config, include siblings replace included
                # keys and add the instance's own (its hart and nodes).
                _replace(loaded, walked)
                return loaded
            return _merge(loaded, walked, base / inc)
        result = {}
        for key, value in obj.items():
            if key == "config" and value == "default" and "ip" in obj:
                repo = Path(__file__).resolve().parents[5]
                catalog = _load(repo / "arch/uvm.toml")
                path = repo / catalog["paths"][obj["ip"]] / "configs/uvm.toml"
                result[key] = _walk_file(path, includes, stack)
            elif key == "param" and "instructions" in obj:
                # A Ball's parameters: its own config, or a design TOML that
                # includes it and replaces some of its values.
                if not isinstance(value, str) or not value.endswith(".toml"):
                    _die(f"{base}: Ball param must name a TOML file, got {value!r}")
                result[key] = _walk_file(
                    (base / value).resolve(), includes, stack, True
                )
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


def _replace(config: dict, value: dict) -> None:
    for key, item in value.items():
        if isinstance(config.get(key), dict) and isinstance(item, dict):
            _replace(config[key], item)
        else:
            config[key] = item


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


def ball2json(path: Path) -> dict[str, Any]:
    """The config of a Ball alone: its default config `path`, <root>/configs/
    default.toml. Its top-level keys are the Ball's parameters, [ball] its
    interface (ports, instructions), [host] the BallDomain that holds it alone;
    [host] values replace those of the framework default host."""
    path = Path(path).resolve()
    includes = [str(path)]
    raw = _load(path)
    for section in ("ball", "host"):
        if section not in raw:
            _die(f"{path}: a Ball's default config needs a [{section}] table")
    host = _walk(raw["host"], path.parent, includes, [str(path)], "host", override=True)
    param = {key: value for key, value in raw.items() if key not in ("ball", "host")}
    param["_file"] = str(path)
    component = path.parents[1].name
    ball = {
        "id": 0,
        "name": component,
        "component": component,
        "port": raw["ball"]["port"],
        "instructions": raw["ball"]["instructions"],
        "param": param,
    }
    return {
        "_file": str(path),
        "includes": includes,
        "ball": {"host": host, "balldomain": {"balls": [ball]}},
        "sims": {"selection": f"ball/{component}/{path.stem}"},
    }


def thread2json(path: Path) -> dict[str, Any]:
    """The config of a thread alone: its default config `path`, <root>/configs/
    default.toml, and the framework's default host, the placement a hart gives
    its one thread."""
    path = Path(path).resolve()
    thread = toml2json(path)
    includes = thread.pop("includes")
    host = _walk_file(_reference(path.parent, "default", "threadhost"), includes, [])
    return {
        "_file": str(path),
        "includes": includes,
        "thread": {"host": host, "config": thread},
        "sims": {"selection": f"thread/{path.parents[1].name}/{path.stem}"},
    }


def _select_hart(data: dict[str, Any], name: str, path: Path) -> None:
    """Keep the first hart of class `name` of the device of `data`, alone on
    the virtual SoC of the hart entry."""
    harts = [hart for hart in data["device"]["harts"] if hart["component"] == name]
    if not harts:
        _die(f"{path}: the device has no hart of class {name!r}")
    hart = harts[0]
    # Alone, the hart is hart 0 of its system and, behind an external Home,
    # its requester node 1.
    hart["hartId"] = 0
    hart["nodeId"] = 1
    data["device"]["harts"] = [hart]
    for peripheral in ("clint", "plic"):
        data["device"][peripheral]["hartIds"] = [hart["hartId"]]
    data["sims"]["targets"] = {"hart": data["sims"]["targets"]["hart"][name]}


def hart2json(path: Path) -> dict[str, Any]:
    """The config of a hart alone: `path`, <root>/sims/default.toml, the
    instance of the hart on the single-hart system of its entry <root>/sims
    (it includes the hart's config, <root>/configs/default.toml, and
    places the instance as a device does), on the framework's default host,
    the peripherals and simulation parameters of that system, whose values its
    [host] table replaces. The entry's
    arch/HartConfig.scala is in the package named by its path."""
    path = Path(path).resolve()
    # As a device config's hart instance, the instance's keys replace those of
    # the hart config it includes and add its own.
    includes: list[str] = []
    hart = _walk_file(path, includes, [], True)
    host = _walk_file(_reference(path.parent, "default", "harthost"), includes, [])
    # [host] of the instance replaces values of the framework host, as the
    # hart's ports need (the DDR AXI ID width of its uncached port).
    if "host" in hart:
        _override(host, hart.pop("host"), str(path))
    repo = Path(__file__).resolve().parents[5]
    entry = path.parent
    # Scala packages drop the hyphens of directory names (multi-rocket is
    # multirocket).
    package = ".".join(part.replace("-", "") for part in entry.relative_to(repo).parts)
    # The hart directory of the entry owns the simulation, even when its config
    # includes another hart's (a design's copy of a prefab hart).
    owner = str(entry.parent / "configs/default.toml")
    device = host["device"]
    device["_file"] = owner
    device["harts"] = [hart]
    sims = host["sims"]
    sims["targets"] = {"hart": {hart["component"]: {"config": f"{package}.HartConfig"}}}
    data = {
        "_file": owner,
        "includes": includes,
        "device": device,
        "sims": sims,
        "workloads": host["workloads"],
    }
    _select_hart(data, hart["component"], path)
    data["sims"]["selection"] = f"hart/{entry.parent.name}/{path.stem}"
    return data


def config2json(path: Path, kind: str, name: str) -> dict[str, Any]:
    """The config of one simulation: config TOML Top `path` includes the design
    and replaces some of its values; `kind` `name` selects the component of the
    design to simulate."""
    path = Path(path).resolve()
    raw = _load(path)
    design = raw.pop("include")
    data = design2json(path.parent / design)
    data["includes"].append(str(path))
    overrides = _walk(raw, path.parent, data["includes"], [str(path)], override=True)
    _override(data, overrides, str(path))
    if kind == "device":
        device = Path(data["_file"]).parents[1].name
        if name != device:
            _die(f"{path}: the device of this config is {device!r}, not {name!r}")
        data["sims"]["targets"] = {kind: data["sims"]["targets"][kind]}
    elif kind == "hart":
        _select_hart(data, name, path)
    else:
        _die(f"{path}: --{kind} simulation is not implemented yet")
    data["sims"]["selection"] = f"{kind}/{name}/{path.stem}"
    return data


def main() -> None:
    if len(sys.argv) != 4:
        _die(f"usage: {sys.argv[0]} CONFIG.toml KIND NAME")
    out = config2json(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
    sys.stdout.write(json.dumps(out, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
