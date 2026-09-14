from __future__ import annotations

import argparse
import fcntl
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path


def _run(
    cmd: list[str],
    *,
    repo: Path,
    cwd: Path | None = None,
    logger: object | None = None,
    task_scope: str | None = None,
    output_prefix: str,
) -> None:
    if logger is None:
        result = subprocess.run(cmd, cwd=cwd or repo)
    else:
        from utils.stream_run import stream_run_logger

        result = stream_run_logger(
            cmd=shlex.join(cmd),
            logger=logger,
            cwd=str(cwd or repo),
            stdout_prefix=output_prefix,
            stderr_prefix=output_prefix,
            task_scope=task_scope,
        )
    if result.returncode != 0:
        detail = result.stderr.strip() if logger is not None else ""
        if not detail and logger is not None:
            lines = result.stdout.splitlines()
            failed = [i for i, line in enumerate(lines) if "FAILED:" in line]
            if failed:
                start = failed[0]
                detail = "\n".join(lines[start:])
            else:
                detail = "\n".join(lines[-20:])
        if detail:
            raise RuntimeError(
                f"command failed ({result.returncode}): {' '.join(cmd)}\n{detail}"
            )
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}")


def compiler_python(repo: str | Path) -> str:
    root = Path(repo).resolve()
    candidates = [root / "result" / "bin" / "python3"]
    path_python = shutil.which("python3")
    if path_python:
        candidates.append(Path(path_python))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        probe = subprocess.run(
            [str(candidate), "-c", "import nanobind"], capture_output=True
        )
        if probe.returncode == 0:
            return str(candidate)
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(f"python3 with nanobind not found (checked: {checked})")


def build_llvm(
    repo: str | Path, *, logger: object | None = None, task_scope: str | None = None
) -> Path:
    root = Path(repo).resolve()
    buddy = root / "compiler" / "thirdparty" / "buddy-mlir"
    llvm_src = buddy / "llvm" / "llvm"
    llvm_build = buddy / "llvm" / "build"
    if not llvm_src.is_dir():
        raise RuntimeError(f"missing LLVM source: {llvm_src}")

    python = compiler_python(root)
    cmake = [
        "cmake",
        "-G",
        "Ninja",
        "-S",
        str(llvm_src),
        "-B",
        str(llvm_build),
        "-DLLVM_ENABLE_PROJECTS=mlir;clang",
        "-DLLVM_ENABLE_RUNTIMES=openmp",
        "-DLLVM_TARGETS_TO_BUILD=host;RISCV",
        "-DLLVM_ENABLE_ASSERTIONS=ON",
        "-DOPENMP_ENABLE_LIBOMPTARGET=OFF",
        "-DLIBOMP_LIBFLAGS=-lrt",
        "-DCMAKE_BUILD_TYPE=RELEASE",
        "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
        f"-DPython3_EXECUTABLE={python}",
        f"-DPython_EXECUTABLE={python}",
    ]
    llvm_build.mkdir(parents=True, exist_ok=True)
    with open(llvm_build / ".bbdev.lock", "a+b") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        if not (llvm_build / "build.ninja").is_file():
            _run(
                cmake,
                repo=root,
                cwd=buddy,
                logger=logger,
                task_scope=task_scope,
                output_prefix="compiler llvm configure",
            )
        _run(
            ["ninja", "-C", str(llvm_build), "-j", str(os.cpu_count() or 1)],
            repo=root,
            cwd=buddy,
            logger=logger,
            task_scope=task_scope,
            output_prefix="compiler llvm build",
        )
    mlir_cmake = llvm_build / "lib" / "cmake" / "mlir"
    if not mlir_cmake.is_dir():
        raise RuntimeError(f"LLVM/MLIR build failed: missing {mlir_cmake}")
    return llvm_build


def compiler_build_dir(repo: str | Path, instance: str) -> Path:
    return (
        Path(repo).resolve()
        / "compiler"
        / "thirdparty"
        / "buddy-mlir"
        / "build"
        / instance
    )


def build_compiler(
    repo: str | Path,
    *,
    chip: str,
    instance: str | None = None,
    chip_pb: str | Path | None = None,
    logger: object | None = None,
    task_scope: str | None = None,
) -> Path:
    root = Path(repo).resolve()
    if not chip:
        raise ValueError("chip is required")
    build_instance = instance or chip
    if not re.fullmatch(r"[A-Za-z0-9_-]+", build_instance):
        raise ValueError(f"invalid compiler build instance: {build_instance}")
    if (instance is None) != (chip_pb is None):
        raise ValueError("compiler variant requires both instance and chip_pb")
    selected_pb = (
        Path(chip_pb).resolve()
        if chip_pb is not None
        else root / "examples" / "chips" / chip / "configs" / "generated" / "chip.pb"
    )
    if not selected_pb.is_file():
        raise RuntimeError(f"missing {selected_pb}; generate the requested chip.pb first")
    buddy = root / "compiler" / "thirdparty" / "buddy-mlir"
    llvm = build_llvm(root, logger=logger, task_scope=task_scope)
    python = compiler_python(root)
    build = compiler_build_dir(root, build_instance)
    build.mkdir(parents=True, exist_ok=True)
    cmake = [
        "cmake",
        "-G",
        "Ninja",
        "-S",
        str(buddy),
        "-B",
        str(build),
        f"-DBUDDY_EXTERNAL_DIALECTS_DIR={root / 'compiler'}",
        f"-DBUCKYBALL_CHIP_PB={selected_pb}",
        f"-DMLIR_DIR={llvm / 'lib' / 'cmake' / 'mlir'}",
        f"-DLLVM_DIR={llvm / 'lib' / 'cmake' / 'llvm'}",
        "-DLLVM_ENABLE_ASSERTIONS=ON",
        "-DCMAKE_BUILD_TYPE=RELEASE",
        "-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON",
        f"-DPython3_EXECUTABLE={python}",
        f"-DPython_EXECUTABLE={python}",
    ]
    _run(
        cmake,
        repo=root,
        logger=logger,
        task_scope=task_scope,
        output_prefix="compiler configure",
    )
    _run(
        [
            "ninja",
            "-C",
            str(build),
            "-j",
            str(os.cpu_count() or 1),
            "buddy-opt",
            "buddy-translate",
            "buddy-llc",
            "rax-pack",
            "python-package-buddy",
            "BuddyMLIRPythonModules",
        ],
        repo=root,
        logger=logger,
        task_scope=task_scope,
        output_prefix="compiler build",
    )
    for tool in ("buddy-opt", "buddy-translate", "buddy-llc", "rax-pack"):
        if not (build / "bin" / tool).is_file():
            raise RuntimeError(f"compiler build failed: missing {build / 'bin' / tool}")
    return build


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a chip compiler instance")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--chip", required=True)
    parser.add_argument("--instance")
    parser.add_argument("--chip-pb", type=Path)
    args = parser.parse_args()
    build_compiler(
        args.repo,
        chip=args.chip,
        instance=args.instance,
        chip_pb=args.chip_pb,
    )


if __name__ == "__main__":
    main()
