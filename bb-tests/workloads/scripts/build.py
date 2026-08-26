from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

_MODELS: dict[str, tuple[str, str]] = {
    "lenet": ("lenet", "buddy-buckyball-lenet-run"),
    "mobilenet": ("mobilenetv3", "buddy-buckyball-mobilenetv3-run"),
    "resnet": ("resnet18", "buddy-buckyball-resnet-run"),
    "yolo": ("yolo26", "buddy-buckyball-yolo26-run"),
    "bert": ("bert", "buddy-buckyball-bert-run"),
    "qwen3": ("qwen3", "buddy-buckyball-qwen3-run"),
    "gemma4": ("gemma4", "buddy-buckyball-gemma4-run"),
    "deepseekr1": ("deepseekr1", "buddy-buckyball-deepseekr1-run"),
    "llama2": ("llama2", "buddy-buckyball-llama2-run"),
    "stable-diffusion": ("stablediffusion", "buddy-buckyball-stable-diffusion-run"),
    "whisper": ("whisper", "buddy-buckyball-whisper-run"),
    "buddynext": ("buddynext", "buddy-buckyball-buddynext-all-run"),
}

_RUSHB: dict[str, dict[str, str]] = {
    "lenet": {
        "bemu": "buddy-buckyball-lenet-rushB-bemu-run",
        "verilator": "buddy-buckyball-lenet-rushB-verilator-run",
    },
    "mobilenet": {
        "bemu": "buddy-buckyball-mobilenetv3-rushB-bemu-run",
        "verilator": "buddy-buckyball-mobilenetv3-rushB-verilator-run",
    },
    "resnet": {
        "bemu": "buddy-buckyball-resnet-rushB-bemu-run",
        "verilator": "buddy-buckyball-resnet-rushB-verilator-run",
    },
    "yolo": {
        "bemu": "buddy-buckyball-yolo26-rushB-bemu-run",
        "verilator": "buddy-buckyball-yolo26-rushB-verilator-run",
    },
    "bert": {
        "bemu": "buddy-buckyball-bert-rushB-bemu-run",
        "verilator": "buddy-buckyball-bert-rushB-verilator-run",
    },
    "qwen3": {
        "bemu": "buddy-buckyball-qwen3-rushB-bemu-run",
        "verilator": "buddy-buckyball-qwen3-rushB-verilator-run",
    },
    "gemma4": {
        "bemu": "buddy-buckyball-gemma4-rushB-bemu-run",
        "verilator": "buddy-buckyball-gemma4-rushB-verilator-run",
    },
    "deepseekr1": {
        "bemu": "buddy-buckyball-deepseekr1-rushB-bemu-run",
        "verilator": "buddy-buckyball-deepseekr1-rushB-verilator-run",
    },
    "llama2": {
        "bemu": "buddy-buckyball-llama2-rushB-bemu-run",
        "verilator": "buddy-buckyball-llama2-rushB-verilator-run",
    },
    "stable-diffusion": {
        "bemu": "buddy-buckyball-stable-diffusion-rushB-bemu-run",
        "verilator": "buddy-buckyball-stable-diffusion-rushB-verilator-run",
    },
    "whisper": {
        "bemu": "buddy-buckyball-whisper-rushB-bemu-run",
        "verilator": "buddy-buckyball-whisper-rushB-verilator-run",
    },
    "buddynext": {
        "bemu": "buddy-buckyball-buddynext-rushB-bemu-run",
        "verilator": "buddy-buckyball-buddynext-rushB-verilator-run",
    },
}


def _repo(raw: str | Path) -> Path:
    root = Path(raw).resolve()
    if not root.is_dir():
        raise ValueError(f"repo does not exist: {root}")
    return root


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    prefix: str,
    env: dict | None = None,
    logger: object | None = None,
    task_scope: str | None = None,
) -> None:
    if logger is None:
        result = subprocess.run(cmd, cwd=cwd, env=env)
    else:
        from utils.stream_run import stream_run_logger

        result = stream_run_logger(
            cmd=shlex.join(cmd),
            logger=logger,
            cwd=str(cwd),
            stdout_prefix=prefix,
            stderr_prefix=prefix,
            task_scope=task_scope,
            env=env,
        )
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}")


def _cmake_defs(repo: Path, chip: str) -> dict[str, str]:
    path = (
        repo
        / "examples"
        / "chips"
        / chip
        / "configs"
        / "generated"
        / "workload"
        / "cmake.defs"
    )
    if not path.is_file():
        raise RuntimeError(f"missing {path}; run bbdev config --install")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if not sep:
            raise RuntimeError(f"bad cmake.defs line in {path}: {line!r}")
        if key in values:
            raise RuntimeError(f"duplicate cmake.defs key {key} in {path}")
        values[key] = value
    required = (
        "BUCKYBALL_WORKLOAD_CHIP",
        "BUCKYBALL_CHIP_PB",
    )
    missing = [k for k in required if k not in values]
    if missing:
        raise RuntimeError(f"{path} missing {missing}")
    return values


_RUSHB_DEFS = (
    "BUCKYBALL_CARGO_TARGET_DIR",
    "BUCKYBALL_RUSHB_BEMU_MANIFEST",
    "BUCKYBALL_RUSHB_BEMU_LIBRARY",
    "BUCKYBALL_RUSHB_VERILATOR_LIBRARY",
)


def _ninja_target(model: str, rushb: str | None) -> tuple[str, str]:
    cmake_model = ""
    ninja_arg = ""
    if model:
        if model not in _MODELS:
            raise ValueError(f"unknown workload model: {model}")
        cmake_model, ninja_arg = _MODELS[model]
        if rushb:
            mapped = _RUSHB.get(model, {}).get(rushb)
            if not mapped:
                raise ValueError(
                    f"no rushB target for model {model!r} backend {rushb!r}"
                )
            ninja_arg = mapped
    elif rushb:
        ninja_arg = f"rushB-{rushb}-workloads-build"
    return cmake_model, ninja_arg


def _require_riscv() -> Path:
    raw = os.environ.get("RISCV", "")
    if not raw:
        raise RuntimeError("RISCV is unset; enter nix develop")
    root = Path(raw)
    if not root.is_dir():
        raise RuntimeError(f"RISCV is not a directory: {root}")
    return root


def _workload_src(repo: Path) -> Path:
    src = repo / "bb-tests" / "workloads"
    cmake = src / "CMakeLists.txt"
    if not cmake.is_file():
        raise RuntimeError(f"missing {cmake}")
    return src


def _workload_build_dir(repo: Path, chip: str) -> Path:
    return repo / "bb-tests" / "workloads" / "build" / chip


def build_workload(
    repo: str | Path,
    chip: str,
    *,
    model: str = "",
    rushb: str | None = None,
    stable: bool = False,
    logger: object | None = None,
    task_scope: str | None = None,
) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", chip):
        raise ValueError(f"invalid chip: {chip}")
    if rushb is not None and rushb not in {"bemu", "verilator"}:
        raise ValueError(f"rushB must be bemu|verilator, got {rushb!r}")

    root = _repo(repo)
    defs = _cmake_defs(root, chip)
    if rushb:
        missing = [k for k in _RUSHB_DEFS if k not in defs]
        if missing:
            raise RuntimeError(
                f"missing {missing} in cmake.defs for rushB; run bbdev config --install"
            )
    compiler_build = root / "compiler" / "thirdparty" / "buddy-mlir" / "build" / chip
    python = shutil.which("python3")
    if not python:
        raise RuntimeError("python3 not in PATH; enter nix develop")

    if rushb == "bemu":
        env = os.environ.copy()
        env["CARGO_TARGET_DIR"] = defs["BUCKYBALL_CARGO_TARGET_DIR"]
        _run(
            [
                "cargo",
                "build",
                "--release",
                "--manifest-path",
                defs["BUCKYBALL_RUSHB_BEMU_MANIFEST"],
                "--lib",
            ],
            cwd=root,
            env=env,
            prefix="workload cargo",
            logger=logger,
            task_scope=task_scope,
        )

    riscv = _require_riscv()
    linux_cc = riscv / "bin" / "riscv64-unknown-linux-gnu-gcc"
    linux_cxx = riscv / "bin" / "riscv64-unknown-linux-gnu-g++"
    if not linux_cc.is_file() or not linux_cxx.is_file():
        raise RuntimeError(f"missing RISC-V linux toolchain under {riscv / 'bin'}")

    src = _workload_src(root)
    build = _workload_build_dir(root, chip)
    cmake_model, ninja_arg = _ninja_target(model.lower(), rushb)
    env = os.environ.copy()
    env["PATH"] = f"{riscv / 'bin'}:{env.get('PATH', '')}"
    env["RISCV"] = str(riscv)
    env["CC"] = str(linux_cc)
    env["CXX"] = str(linux_cxx)
    env["BUDDY_MLIR_BUILD_DIR"] = str(compiler_build)

    build.mkdir(parents=True, exist_ok=True)

    cmake_args = [
        "cmake",
        "-G",
        "Ninja",
        "-S",
        str(src),
        "-B",
        str(build),
        f"-DBUCKYBALL_STABLE={'ON' if stable else 'OFF'}",
        f"-DPython3_EXECUTABLE={python}",
        f"-DCMAKE_C_COMPILER={linux_cc}",
        f"-DCMAKE_CXX_COMPILER={linux_cxx}",
    ]
    for key, value in defs.items():
        if key in _RUSHB_DEFS and not rushb:
            continue
        cmake_args.append(f"-D{key}={value}")
    if cmake_model:
        cmake_args.extend(["-DMODEL=" + cmake_model, "-DARCH=buckyball"])

    _run(
        cmake_args,
        cwd=root,
        env=env,
        prefix="workload configure",
        logger=logger,
        task_scope=task_scope,
    )
    ninja = ["ninja", "-C", str(build), f"-j{os.cpu_count() or 1}"]
    if ninja_arg:
        ninja.append(ninja_arg)
    _run(
        ninja,
        cwd=root,
        env=env,
        prefix="workload build",
        logger=logger,
        task_scope=task_scope,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chip workloads")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--chip", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--rushb", choices=("bemu", "verilator"))
    parser.add_argument("--stable", action="store_true")
    args = parser.parse_args()
    build_workload(
        args.repo,
        args.chip,
        model=args.model,
        rushb=args.rushb,
        stable=args.stable,
    )


if __name__ == "__main__":
    main()
