import json
import importlib.util
import re
import tomllib
from datetime import datetime
from pathlib import Path

from utils.path import log_dir


def model_run_commands(repo: str, params: dict):
    unknown = params.keys() - {
        "chip",
        "model",
        "reuse-simulator",
        "itrace",
        "mtrace",
        "firmware",
        "guest-memory-mib",
        "load-manifest",
    }
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
        python = root / "result/bin/python3"
    else:
        raise ValueError(f"Unknown model execution kind: {kind}")
    firmware = None
    endpoint = layout["execution"].get("p2e", {})
    if endpoint.get("task_runtime") == "ant" or (
        kind == "native" and layout["execution"]["input"] == "resource"
    ):
        if kind != "native" or endpoint.get("kind") != "native":
            raise ValueError("prepared model execution must run inside the Linux guest")
        binary = f"bebop-chip-{chip}"
        memory = params.get("guest-memory-mib", settings["memory_mib"])
        if memory is not None:
            if isinstance(memory, bool) or not str(memory).isdigit() or int(memory) < 1:
                raise ValueError("guest-memory-mib must be a positive integer")
            memory = int(memory)
        spec = importlib.util.spec_from_file_location(
            "buckyball_model_guest", root / "stack/serving/guest.py"
        )
        guest = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guest)
        firmware, guest_memory, _ = guest.firmware_profile(
            root, chip, model, params.get("firmware"), memory
        )
    elif any(
        key in params for key in ("firmware", "guest-memory-mib", "load-manifest")
    ):
        raise ValueError("firmware options require a packaged Linux guest model")
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
    if firmware is not None:
        commands[-1].extend(
            ["--firmware", str(firmware), "--guest-memory-mib", str(guest_memory)]
        )
        if firmware.name.endswith("-ddr.elf"):
            manifest = firmware.with_suffix(".load.json")
            if (
                "load-manifest" in params
                and Path(params["load-manifest"]).resolve() != manifest.resolve()
            ):
                raise ValueError(
                    "load-manifest does not belong to the requested firmware"
                )
            if not manifest.is_file():
                raise ValueError(f"MODEL_DDR load manifest missing: {manifest}")
            commands[-1].extend(["--load-manifest", str(manifest)])
        elif "load-manifest" in params:
            raise ValueError("load-manifest requires a MODEL_DDR firmware")
    if params.get("reuse-simulator", False):
        simulator = target / "release" / binary
        if not simulator.is_file():
            raise ValueError(f"Reusable model simulator missing: {simulator}")
        commands = commands[1:]
    commands[-1].extend(
        f"--{flag}" for flag in ("itrace", "mtrace") if params.get(flag, False)
    )
    return commands, log
