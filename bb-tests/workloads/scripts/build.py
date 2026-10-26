from __future__ import annotations
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    prefix: str,
    logger: object | None = None,
    task_scope: str | None = None,
) -> None:
    if logger is None:
        result = subprocess.run(cmd, cwd=cwd)
    else:
        from utils.stream_run import stream_run_logger

        result = stream_run_logger(
            cmd=shlex.join(cmd),
            logger=logger,
            cwd=str(cwd),
            stdout_prefix=prefix,
            stderr_prefix=prefix,
            task_scope=task_scope,
        )
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}")


def build_workload(
    repo: str | Path,
    design: str,
    *,
    selection: str,
    ctest: bool = False,
    mlirtest: bool = False,
    soctest: bool = False,
    logger: object | None = None,
    task_scope: str | None = None,
) -> None:
    if not re.fullmatch("[A-Za-z0-9_-]+", design):
        raise ValueError(f"invalid design: {design}")
    if sum((ctest, mlirtest, soctest)) > 1:
        raise ValueError("--ctest, --mlirtest and --soctest are mutually exclusive")
    root = Path(repo).resolve()
    design_pb = root / "examples/designs" / design / "output/generated/design.pb"
    sys.path.insert(0, str(root))
    from stack.scripts.design import load

    target = load(design_pb).simulations[selection]
    if not target.directory:
        raise ValueError(f"{design}: unregistered simulation entry {selection}")
    defs = {
        "BUCKYBALL_WORKLOAD_DESIGN": design,
        "BUCKYBALL_DESIGN_PB": str(design_pb),
    }
    defs["BUCKYBALL_WORKLOAD_SCOPE"] = (
        "ctest"
        if ctest
        else "mlirtest" if mlirtest else "soctest" if soctest else "all"
    )
    defs["BUCKYBALL_SIM_TARGET"] = selection
    defs["BUCKYBALL_DESIGN_WORKLOAD_DIR"] = str(root / target.workloads)
    defs["BUCKYBALL_WORKLOAD_OUTPUT"] = str(
        root / "examples/designs" / design / "output/sims" / selection / "workloads"
    )
    tools = {}
    for name in (
        "cmake",
        "riscv64-unknown-elf-gcc",
        "riscv64-unknown-elf-g++",
        "riscv64-unknown-elf-objcopy",
        "riscv64-unknown-linux-gnu-gcc",
        "riscv64-unknown-linux-gnu-g++",
    ):
        executable = shutil.which(name)
        if executable is None:
            raise FileNotFoundError(f"{name} is required; enter nix develop")
        tools[name] = str(Path(executable).absolute())
    source = root / "bb-tests" / "workloads"
    build = (
        root
        / "examples/designs"
        / design
        / "output/sims"
        / selection
        / "workloads-build"
    )
    target = (
        "sync-ctest-bin"
        if ctest
        else (
            "sync-mlirtest-bin"
            if mlirtest
            else "sync-soctest-bin" if soctest else "sync-bin"
        )
    )
    build.mkdir(parents=True, exist_ok=True)
    cmake_args = [
        tools["cmake"],
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_C_FLAGS_RELEASE=-O2 -UNDEBUG",
        "-DCMAKE_CXX_FLAGS_RELEASE=-O2 -UNDEBUG",
        "-S",
        str(source),
        "-B",
        str(build),
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DCMAKE_C_COMPILER={tools['riscv64-unknown-linux-gnu-gcc']}",
        f"-DCMAKE_CXX_COMPILER={tools['riscv64-unknown-linux-gnu-g++']}",
    ]
    for variable, executable in (
        ("ELF_CC", "riscv64-unknown-elf-gcc"),
        ("ELF_CXX", "riscv64-unknown-elf-g++"),
        ("ELF_OBJCOPY", "riscv64-unknown-elf-objcopy"),
    ):
        cmake_args.append(f"-D{variable}={tools[executable]}")
    for key, value in defs.items():
        cmake_args.append(f"-D{key}={value}")
    _run(
        cmake_args,
        cwd=root,
        prefix="workload configure",
        logger=logger,
        task_scope=task_scope,
    )
    _run(
        [tools["cmake"], "--build", str(build), "--parallel", "1", "--target", target],
        cwd=root,
        prefix="workload build",
        logger=logger,
        task_scope=task_scope,
    )
