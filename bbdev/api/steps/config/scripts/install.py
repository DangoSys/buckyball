#!/usr/bin/env python3
"""Install the design.pb of one simulation: config TOML Top CONFIG (relative to
examples/designs) with component KIND NAME selected. Every build system reads
that design.pb; the selection names its output directory."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

scripts = Path(__file__).resolve().parent


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, scripts / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _protoc(repo: Path, schemas: list[Path]) -> None:
    """The Python modules of `schemas`, next to this script."""
    with tempfile.TemporaryDirectory(prefix="bbdev-proto-") as out:
        subprocess.run(
            [
                "protoc",
                *[
                    f"-I{directory}"
                    for directory in sorted({p.parent for p in schemas})
                ],
                f"--python_out={out}",
                *map(str, schemas),
            ],
            check=True,
        )
        for source in Path(out).glob("*_pb2.py"):
            content = source.read_bytes()
            destination = scripts / source.name
            if not destination.is_file() or destination.read_bytes() != content:
                destination.write_bytes(content)


def ball_root(repo: Path, name: str) -> Path:
    """A Ball named alone is a prefab Ball; any other is a path relative to
    examples/designs, <design>/balls/<ball>."""
    root = (
        repo / "examples/prefab/balls" / name
        if "/" not in name
        else repo / "examples/designs" / name
    )
    if not (root / "configs/default.toml").is_file():
        raise FileNotFoundError(f"no Ball {name!r}: {root}/configs/default.toml")
    return root


def install_ball(repo: Path, name: str) -> str:
    """The design.pb of Ball `name` alone, from its default config, under
    <ball>/output/sims/<selection>; return the selection."""
    toml2json = _load_script("1_toml2json.py")
    root = ball_root(repo, name)
    data = toml2json.ball2json(root / "configs/default.toml")
    selection = data["sims"]["selection"]
    generated = root / "output/sims" / selection / "generated"
    config_json = generated / "config/config.json"
    config_json.parent.mkdir(parents=True, exist_ok=True)
    config_json.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    _protoc(
        repo,
        [
            scripts / "proto/design.proto",
            scripts / "proto/hardware.proto",
            scripts / "proto/model.proto",
            *sorted((root / "configs").glob("*_params.proto")),
        ],
    )
    sys.path.insert(0, str(scripts))
    json2proto = _load_script("3_json2proto.py")
    json2proto.write_ball_pb(config_json, generated / "design.pb", repo)
    return selection


def thread_root(repo: Path, name: str) -> Path:
    """A thread named alone is a prefab thread; any other is a path relative
    to examples/designs, <design>/threads/<thread>."""
    root = (
        repo / "examples/prefab/threads" / name
        if "/" not in name
        else repo / "examples/designs" / name
    )
    if not (root / "configs/default.toml").is_file():
        raise FileNotFoundError(f"no thread {name!r}: {root}/configs/default.toml")
    return root


def install_thread(repo: Path, name: str) -> str:
    """The design.pb of thread `name` alone, from its default config, under
    <thread>/output/sims/<selection>; return the selection."""
    toml2json = _load_script("1_toml2json.py")
    root = thread_root(repo, name)
    data = toml2json.thread2json(root / "configs/default.toml")
    selection = data["sims"]["selection"]
    generated = root / "output/sims" / selection / "generated"
    config_json = generated / "config/config.json"
    config_json.parent.mkdir(parents=True, exist_ok=True)
    config_json.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    sys.path.insert(0, str(scripts))
    derive = _load_script("2_parameter_derivation.py")
    from ball_normalize import normalize_ball_domain

    domain = normalize_ball_domain(data["thread"]["config"])
    balls = [] if domain is None else domain["balls"]
    _protoc(
        repo,
        [
            scripts / "proto/design.proto",
            scripts / "proto/hardware.proto",
            scripts / "proto/model.proto",
            *sorted(
                {
                    proto
                    for ball in balls
                    for proto in (derive.ball_root(ball) / "configs").glob(
                        "*_params.proto"
                    )
                }
            ),
        ],
    )
    json2proto = _load_script("3_json2proto.py")
    json2proto.write_thread_alone_pb(config_json, generated / "design.pb", repo)
    return selection


def hart_root(repo: Path, name: str) -> Path:
    """A hart named alone is a prefab hart; any other is a path relative to
    examples/designs, <design>/harts/<hart>."""
    root = (
        repo / "examples/prefab/harts" / name
        if "/" not in name
        else repo / "examples/designs" / name
    )
    if not (root / "sims/hart/default.toml").is_file():
        raise FileNotFoundError(f"no hart {name!r}: {root}/sims/hart/default.toml")
    return root


def install_hart(repo: Path, name: str) -> str:
    """The design.pb of hart `name` alone, its instance <hart>/sims/hart/
    default.toml on the framework's single-hart system, under <hart>/output/
    sims/<selection>; return the selection."""
    toml2json = _load_script("1_toml2json.py")
    derive = _load_script("2_parameter_derivation.py")
    root = hart_root(repo, name)
    data = toml2json.hart2json(root / "sims/hart/default.toml")
    selection = data["sims"]["selection"]
    generated = root / "output/sims" / selection / "generated"
    config_json = generated / "config/config.json"
    config_json.parent.mkdir(parents=True, exist_ok=True)
    config_json.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    derived_json = config_json.with_name("derived.json")
    derive.write_derived(
        data, repo, str(root.relative_to(repo / "examples")), derived_json
    )
    _protoc(
        repo,
        [
            scripts / "proto/design.proto",
            scripts / "proto/hardware.proto",
            scripts / "proto/model.proto",
            *derive.ball_protos(data["device"]),
            *sorted(repo.glob("examples/prefab/harts/*/configs/*_params.proto")),
        ],
    )
    sys.path.insert(0, str(scripts))
    json2proto = _load_script("3_json2proto.py")
    json2proto.write_design_pb(config_json, derived_json, generated / "design.pb", repo)
    return selection


def install(repo: Path, config: str, kind: str, name: str) -> str:
    """Write output/sims/<selection>/generated/design.pb of the design of
    `config` and return the selection, <kind>/<name>/<config name>. A Ball
    without a config (`config` empty) is installed alone from its default
    config."""
    # Generated protobuf modules must not share a long-lived server's descriptor pool.
    if __name__ != "__main__":
        return subprocess.check_output(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                str(repo),
                config,
                kind,
                name,
            ],
            text=True,
        ).strip()
    repo = Path(repo).resolve()
    if kind == "ball" and not config:
        return install_ball(repo, name)
    if kind == "thread" and not config:
        return install_thread(repo, name)
    if kind == "hart" and not config:
        return install_hart(repo, name)
    design = Path(config).parts[0]
    toml2json = _load_script("1_toml2json.py")
    derive = _load_script("2_parameter_derivation.py")
    data = toml2json.config2json(repo / "examples/designs" / config, kind, name)
    selection = data["sims"]["selection"]
    generated = (
        repo / "examples/designs" / design / "output/sims" / selection / "generated"
    )
    config_json = generated / "config/config.json"
    config_json.parent.mkdir(parents=True, exist_ok=True)
    config_json.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    derived_json = config_json.with_name("derived.json")
    derive.write_derived(data, repo, design, derived_json)

    # The Python modules of design.proto, hardware.proto, model.proto and the
    # parameter schemas of the Balls the simulation instantiates.
    schemas = [
        scripts / "proto/design.proto",
        scripts / "proto/hardware.proto",
        scripts / "proto/model.proto",
        *derive.ball_protos(data["device"]),
        *sorted(repo.glob("examples/prefab/harts/*/configs/*_params.proto")),
    ]
    _protoc(repo, schemas)

    sys.path.insert(0, str(scripts))
    json2proto = _load_script("3_json2proto.py")
    json2proto.write_design_pb(config_json, derived_json, generated / "design.pb", repo)
    return selection


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(f"usage: {sys.argv[0]} REPO CONFIG KIND NAME")
    print(install(Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]))


if __name__ == "__main__":
    main()
