from __future__ import annotations

import argparse
import fcntl
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def _run(
    cmd: list[str],
    *,
    repo: Path,
    logger: object | None = None,
    task_scope: str | None = None,
    output_prefix: str,
) -> None:
    cmd = ["nix", "develop", str(repo), "-c", *cmd]
    if logger is None:
        result = subprocess.run(cmd, cwd=repo)
    else:
        from utils.stream_run import stream_run_logger

        result = stream_run_logger(
            cmd=shlex.join(cmd),
            logger=logger,
            cwd=str(repo),
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
    buddy = root / "stack" / "thirdparty" / "buddy-mlir"
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
                logger=logger,
                task_scope=task_scope,
                output_prefix="compiler llvm configure",
            )
        _run(
            ["ninja", "-C", str(llvm_build), "-j", str(os.cpu_count() or 1)],
            repo=root,
            logger=logger,
            task_scope=task_scope,
            output_prefix="compiler llvm build",
        )
    mlir_cmake = llvm_build / "lib" / "cmake" / "mlir"
    if not mlir_cmake.is_dir():
        raise RuntimeError(f"LLVM/MLIR build failed: missing {mlir_cmake}")
    return llvm_build


def build_buddy(
    repo: str | Path, *, logger: object | None = None, task_scope: str | None = None
) -> Path:
    """The one buddy-mlir build every stack's compiler plugin is loaded into."""
    root = Path(repo).resolve()
    buddy = root / "stack/thirdparty/buddy-mlir"
    llvm = build_llvm(root, logger=logger, task_scope=task_scope)
    python = compiler_python(root)
    build = buddy / "build"
    cmake = [
        "cmake",
        "-G",
        "Ninja",
        "-S",
        str(buddy),
        "-B",
        str(build),
        f"-DMLIR_DIR={llvm / 'lib' / 'cmake' / 'mlir'}",
        f"-DLLVM_DIR={llvm / 'lib' / 'cmake' / 'llvm'}",
        "-DLLVM_ENABLE_ASSERTIONS=ON",
        "-DCMAKE_BUILD_TYPE=RELEASE",
        "-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON",
        f"-DPython3_EXECUTABLE={python}",
        f"-DPython_EXECUTABLE={python}",
    ]
    with open(buddy / ".bbdev.lock", "a+b") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        _run(
            cmake,
            repo=root,
            logger=logger,
            task_scope=task_scope,
            output_prefix="buddy-mlir configure",
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
            output_prefix="buddy-mlir build",
        )
    return build


def _design(repo: Path, design: str):
    sys.path.insert(0, str(repo))
    from stack.scripts import design as design_config

    return design_config, design_config.load(
        repo / "examples/designs" / design / "output/generated/design.pb"
    )


def build_design(
    repo: str | Path,
    *,
    design: str,
    selection: str,
    logger: object | None = None,
    task_scope: str | None = None,
) -> None:
    """Build the software stack of one simulation entry of `design`.

    sims/<selection>/stack/CMakeLists.txt is the top of that build: it lists the
    thread, core and device stacks the entry needs and builds each of them.
    """
    root = Path(repo).resolve()
    design_config, config = _design(root, design)
    if selection not in config.simulations:
        raise ValueError(f"{design}: unregistered simulation entry {selection}")
    build = design_config.stack_build(config, selection)
    design_config.generate(config, build / "generated", selection)
    build_buddy(root, logger=logger, task_scope=task_scope)
    _run(
        [
            "cmake",
            "-G",
            "Ninja",
            "-S",
            config.simulations[selection].stack,
            "-B",
            str(build),
            f"-DDESIGN_CMAKE={build / 'generated/design.cmake'}",
        ],
        repo=root,
        logger=logger,
        task_scope=task_scope,
        output_prefix=f"stack {selection} configure",
    )
    _run(
        ["ninja", "-C", str(build), "-j", str(os.cpu_count() or 1)],
        repo=root,
        logger=logger,
        task_scope=task_scope,
        output_prefix=f"stack {selection} build",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the stacks of a design")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    build_design(args.repo, design=args.design, selection=args.target)


if __name__ == "__main__":
    main()
