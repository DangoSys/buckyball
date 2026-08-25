from __future__ import annotations

import os
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
            detail = "\n".join(result.stdout.splitlines()[-20:])
        if detail:
            raise RuntimeError(
                f"command failed ({result.returncode}): {' '.join(cmd)}\n{detail}"
            )
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}")


def compiler_python() -> str:
    python = shutil.which("python3")
    if not python:
        raise RuntimeError("python3 not in PATH; enter nix develop")
    probe = subprocess.run([python, "-c", "import nanobind"], capture_output=True)
    if probe.returncode != 0:
        raise RuntimeError(f"python3 lacks nanobind ({python}); enter nix develop")
    return python


def build_llvm(
    repo: str | Path, *, logger: object | None = None, task_scope: str | None = None
) -> Path:
    root = _repo(repo)
    buddy = root / "compiler" / "thirdparty" / "buddy-mlir"
    llvm_src = buddy / "llvm" / "llvm"
    llvm_build = buddy / "llvm" / "build"
    if not llvm_src.is_dir():
        raise RuntimeError(f"missing LLVM source: {llvm_src}")

    python = compiler_python()
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


def compiler_build_dir(repo: str | Path, chip: str) -> Path:
    return _repo(repo) / "compiler" / "thirdparty" / "buddy-mlir" / "build" / chip


def build_compiler(
    repo: str | Path,
    *,
    chip: str,
    logger: object | None = None,
    task_scope: str | None = None,
) -> Path:
    root = _repo(repo)
    if not chip:
        raise ValueError("chip is required")
    chip_pb = root / "examples" / "chips" / chip / "configs" / "generated" / "chip.pb"
    if not chip_pb.is_file():
        raise RuntimeError(f"missing {chip_pb}; run bbdev config --install")
    buddy = root / "compiler" / "thirdparty" / "buddy-mlir"
    llvm = build_llvm(root, logger=logger, task_scope=task_scope)
    python = compiler_python()
    build = compiler_build_dir(root, chip)
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
        f"-DBUCKYBALL_CHIP_PB={chip_pb}",
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
            "python-package-buddy",
            "BuddyMLIRPythonModules",
        ],
        repo=root,
        logger=logger,
        task_scope=task_scope,
        output_prefix="compiler build",
    )
    for tool in ("buddy-opt", "buddy-translate", "buddy-llc"):
        if not (build / "bin" / tool).is_file():
            raise RuntimeError(f"compiler build failed: missing {build / 'bin' / tool}")
    return build
