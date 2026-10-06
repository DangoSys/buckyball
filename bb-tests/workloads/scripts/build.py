from __future__ import annotations
import argparse
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path


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
    required = ("BUCKYBALL_WORKLOAD_CHIP", "BUCKYBALL_CHIP_PB")
    missing = [k for k in required if k not in values]
    if missing:
        raise RuntimeError(f"{path} missing {missing}")
    return values


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


def _workload_build_dir(repo: Path, instance: str) -> Path:
    return repo / "bb-tests" / "workloads" / "build" / instance


def build_workload(
    repo: str | Path,
    chip: str,
    *,
    instance: str | None = None,
    chip_pb: str | Path | None = None,
    ctest: bool = False,
    mlirtest: bool = False,
    stable: bool = False,
    logger: object | None = None,
    task_scope: str | None = None,
) -> None:
    if not re.fullmatch("[A-Za-z0-9_-]+", chip):
        raise ValueError(f"invalid chip: {chip}")
    build_instance = instance or chip
    if not re.fullmatch("[A-Za-z0-9_-]+", build_instance):
        raise ValueError(f"invalid workload build instance: {build_instance}")
    if (instance is None) != (chip_pb is None):
        raise ValueError("workload variant requires both instance and chip_pb")
    if ctest and mlirtest:
        raise ValueError("--ctest and --mlirtest cannot be used together")
    root = _repo(repo)
    defs = _cmake_defs(root, chip)
    if chip_pb is not None:
        selected_pb = Path(chip_pb).resolve()
        if not selected_pb.is_file():
            raise RuntimeError(f"missing {selected_pb}")
        defs["BUCKYBALL_WORKLOAD_CHIP"] = build_instance
        defs["BUCKYBALL_WORKLOAD_SOURCE_CHIP"] = chip
        defs["BUCKYBALL_CHIP_PB"] = str(selected_pb)
    compiler_build = (
        root
        / "stack"
        / "compiler"
        / "thirdparty"
        / "buddy-mlir"
        / "build"
        / build_instance
    )
    riscv = _require_riscv()
    project_python = riscv / "bin" / "python3"
    python = (
        str(project_python) if project_python.is_file() else shutil.which("python3")
    )
    if not python:
        raise RuntimeError("python3 not in PATH; enter nix develop")
    linux_cc = riscv / "bin" / "riscv64-unknown-linux-gnu-gcc"
    linux_cxx = riscv / "bin" / "riscv64-unknown-linux-gnu-g++"
    if not linux_cc.is_file() or not linux_cxx.is_file():
        raise RuntimeError(f"missing RISC-V linux toolchain under {riscv / 'bin'}")
    src = _workload_src(root)
    build = _workload_build_dir(root, build_instance)
    ninja_arg = "sync-ctest-bin" if ctest else "sync-mlirtest-bin" if mlirtest else ""
    env = os.environ.copy()
    env["PATH"] = f"{riscv / 'bin'}:{env.get('PATH', '')}"
    env["RISCV"] = str(riscv)
    env["BUDDY_MLIR_BUILD_DIR"] = str(compiler_build)
    env["CC"] = str(linux_cc)
    env["CXX"] = str(linux_cxx)
    if not ctest:
        _run(
            [
                "cmake",
                "--build",
                str(compiler_build),
                "--target",
                "buddy-opt",
                "buddy-translate",
            ],
            cwd=root,
            env=env,
            prefix="workload compiler",
            logger=logger,
            task_scope=task_scope,
        )
    build.mkdir(parents=True, exist_ok=True)
    cmake_args = [
        "cmake",
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_C_FLAGS_RELEASE=-O2 -UNDEBUG",
        "-DCMAKE_CXX_FLAGS_RELEASE=-O2 -UNDEBUG",
        "-S",
        str(src),
        "-B",
        str(build),
        f"-DBUCKYBALL_STABLE={('ON' if stable else 'OFF')}",
        f"-DPython3_EXECUTABLE={python}",
        f"-DCMAKE_C_COMPILER={linux_cc}",
        f"-DCMAKE_CXX_COMPILER={linux_cxx}",
        f"-DBUCKYBALL_CTEST_ONLY={'ON' if ctest else 'OFF'}",
    ]
    for key, value in defs.items():
        cmake_args.append(f"-D{key}={value}")
    _run(
        cmake_args,
        cwd=root,
        env=env,
        prefix="workload configure",
        logger=logger,
        task_scope=task_scope,
    )
    ninja = ["ninja", "-C", str(build), f"-j{1}"]
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
    parser.add_argument("--instance")
    parser.add_argument("--chip-pb", type=Path)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--ctest", action="store_true")
    scope.add_argument("--mlirtest", action="store_true")
    parser.add_argument("--stable", action="store_true")
    args = parser.parse_args()
    build_workload(
        args.repo,
        args.chip,
        instance=args.instance,
        chip_pb=args.chip_pb,
        ctest=args.ctest,
        mlirtest=args.mlirtest,
        stable=args.stable,
    )


if __name__ == "__main__":
    main()
