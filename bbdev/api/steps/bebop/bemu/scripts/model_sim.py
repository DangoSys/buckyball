import json
import re
import tomllib
from datetime import datetime
from pathlib import Path

from utils.path import log_dir


def model_run_commands(repo: str, params: dict):
    unknown = params.keys() - {"chip", "model", "reuse-simulator", "itrace", "mtrace"}
    if unknown:
        raise ValueError(f"Unknown model simulation parameters: {sorted(unknown)}")
    for key in ("chip", "model"):
        if not isinstance(params.get(key), str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+", params[key]
        ):
            raise ValueError(f"Invalid or missing {key}")
    root = Path(repo)
    if not isinstance(params.get("reuse-simulator", False), bool):
        raise ValueError("reuse-simulator must be a boolean")
    for flag in ("itrace", "mtrace"):
        if not isinstance(params.get(flag, False), bool):
            raise ValueError(f"{flag} must be a boolean")
    chip, model = params["chip"], params["model"]
    config = root / "examples/models" / chip / model / "configs/running-param.toml"
    build = root / "stack/models/build" / chip / model
    packages = list((build / "artifact").glob("*.rax"))
    if len(packages) != 1:
        raise ValueError(
            f"Expected one built RAX under {build / 'artifact'}; build the model first"
        )
    layout = json.loads((build / "artifact/layout.json").read_text())
    kind = layout["execution"]["kind"]
    settings = tomllib.loads(config.read_text())
    manifest = root / "examples/chips" / chip / "configs/generated/bemu/Cargo.toml"
    if kind == "native" and "tile_indices" not in settings:
        binary = "bebop-bemu"
        python = root / "result/bin/python3"
    elif kind == "python" or (kind == "native" and "tile_indices" in settings):
        binary = f"bebop-chip-{chip}"
        python = root / ("result/bin/python3" if kind == "native" else "stack/serving/.venv/bin/python")
    else:
        raise ValueError(f"Unknown model execution kind: {kind}")
    target = root / "bebop/target" / chip
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log = Path(log_dir(repo, chip, "verilog", stamp, "bemu", model))
    commands = [
        [
            "cargo",
            "build",
            "--release",
            "--config",
            'build.rustflags=["-C", "target-cpu=native"]',
            "--config",
            'profile.release.lto="thin"',
            "--config",
            "profile.release.codegen-units=1",
            "--manifest-path",
            str(manifest),
            "--target-dir",
            str(target),
            "--bin",
            binary,
        ],
        [
            str(python),
            "-u",
            "-m",
            "stack.serving.cli",
            "--model",
            str(packages[0]),
            "--loader",
            str(build / "loader/load-model"),
            "--simulator",
            str(target / "release" / binary),
            "--log-dir",
            str(log),
            "--run-config",
            str(config),
        ],
    ]
    if params.get("reuse-simulator", False):
        simulator = target / "release" / binary
        if not simulator.is_file():
            raise ValueError(f"Reusable model simulator missing: {simulator}")
        commands = commands[1:]
    commands[-1].extend(f"--{flag}" for flag in ("itrace", "mtrace") if params.get(flag, False))
    return commands, log
