from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from utils.path import bebop_cargo_env, workload_build_dir

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


def _repo(bbdir: str) -> Path:
    root = Path(bbdir).resolve()
    if not root.is_dir():
        raise ValueError(f"repo does not exist: {root}")
    return root


def _configparse(bbdir: str) -> Path:
    path = _repo(bbdir) / "bazel" / "configparse"
    if not path.is_dir():
        raise ValueError(f"missing configparse: {path}")
    return path


def _pythonpath(bbdir: str) -> dict[str, str]:
    env = os.environ.copy()
    parse = _configparse(bbdir)
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{parse}:{prev}" if prev else str(parse)
    return env


def _run(
    cmd: list[str],
    *,
    bbdir: str,
    cwd: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> None:
    env = _pythonpath(bbdir)
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(cmd, cwd=cwd or _repo(bbdir), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}")


def _require_riscv() -> Path:
    raw = os.environ.get("RISCV", "")
    if not raw:
        raise RuntimeError("RISCV is unset; enter nix develop")
    root = Path(raw)
    if not root.is_dir():
        raise RuntimeError(f"RISCV is not a directory: {root}")
    return root


def install_bundle(bbdir: str, chip: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", chip):
        raise ValueError(f"invalid chip: {chip}")
    script = _configparse(bbdir) / "chip_bundle.py"
    _run([sys.executable, str(script), "--repo", str(_repo(bbdir)), "--chip", chip, "--all"], bbdir=bbdir)


def _compiler_core(bbdir: str, chip: str | None, core: str | None) -> str:
    if core:
        return core
    if not chip:
        raise ValueError("chip or core is required")
    sys.path.insert(0, str(_configparse(bbdir)))
    from chip_bundle import chip_index  # type: ignore

    picked = chip_index(_repo(bbdir), chip).get("compilerCore", "")
    if not isinstance(picked, str) or not picked:
        raise ValueError(f"chip {chip} has no compilerCore")
    return picked


def build_compiler(bbdir: str, *, chip: str | None = None, core: str | None = None) -> None:
    root = _repo(bbdir)
    picked = _compiler_core(bbdir, chip, core)
    buddy = root / "compiler" / "thirdparty" / "buddy-mlir"
    dialects = root / "examples" / "cores" / picked / "compiler"
    llvm = buddy / "llvm" / "build"
    build_root = buddy / "build"
    build = build_root / "cores" / picked
    if not (dialects / "CMakeLists.txt").is_file():
        raise RuntimeError(f"missing compiler package: {dialects / 'CMakeLists.txt'}")
    if not (llvm / "lib" / "cmake" / "mlir").is_dir():
        raise RuntimeError(f"missing LLVM/MLIR cmake at {llvm}")
    python = sys.executable
    py_include = subprocess.check_output(
        [python, "-c", "import sysconfig; print(sysconfig.get_path('include'))"],
        text=True,
    ).strip()
    py_lib = subprocess.check_output(
        [
            python,
            "-c",
            "import sysconfig; print(sysconfig.get_config_var('LIBDIR') + '/' + sysconfig.get_config_var('LDLIBRARY'))",
        ],
        text=True,
    ).strip()
    cache = llvm / "CMakeCache.txt"
    nanobind = subprocess.check_output(
        ["sed", "-n", r"s/^NB_DIR:INTERNAL=\(.*\)$/\1\/cmake/p", str(cache)],
        text=True,
    ).strip()
    if not (Path(nanobind) / "nanobind-config.cmake").is_file():
        raise RuntimeError(f"missing nanobind cmake package: {nanobind}")
    build.mkdir(parents=True, exist_ok=True)
    cmake = [
        "cmake",
        "-G",
        "Ninja",
        "-S",
        str(buddy),
        "-B",
        str(build),
        f"-DBUDDY_EXTERNAL_DIALECTS_DIR={dialects}",
        f"-DMLIR_DIR={llvm / 'lib' / 'cmake' / 'mlir'}",
        f"-DLLVM_DIR={llvm / 'lib' / 'cmake' / 'llvm'}",
        "-DLLVM_ENABLE_ASSERTIONS=ON",
        "-DCMAKE_BUILD_TYPE=RELEASE",
        "-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON",
        f"-Dnanobind_DIR={nanobind}",
        f"-DPython3_EXECUTABLE={python}",
        f"-DPython3_INCLUDE_DIR={py_include}",
        f"-DPython3_LIBRARY={py_lib}",
        f"-DPython_EXECUTABLE={python}",
        f"-DPython_INCLUDE_DIR={py_include}",
        f"-DPython_LIBRARY={py_lib}",
    ]
    _run(cmake, bbdir=bbdir)
    _run(
        ["ninja", "-C", str(build), "-j", str(os.cpu_count() or 1), "buddy-opt", "buddy-translate", "buddy-llc", "python-package-buddy", "BuddyMLIRPythonModules"],
        bbdir=bbdir,
    )
    public_bin = build_root / "bin"
    public_bin.mkdir(parents=True, exist_ok=True)
    for tool in ("buddy-opt", "buddy-translate", "buddy-llc"):
        link = public_bin / f"{tool}-{picked}"
        if link.is_symlink():
            link.unlink()
        link.symlink_to(Path("..") / "cores" / picked / "bin" / tool)


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
                raise ValueError(f"no rushB target for model {model!r} backend {rushb!r}")
            ninja_arg = mapped
    elif rushb:
        ninja_arg = f"rushB-{rushb}-workloads-build"
    return cmake_model, ninja_arg


def _workload_core(bbdir: str, chip: str) -> str:
    defs_py = _configparse(bbdir) / "workload_cmake_defs.py"
    out = subprocess.check_output(
        [sys.executable, str(defs_py), "--repo", str(_repo(bbdir)), "--chip", chip],
        text=True,
        env=_pythonpath(bbdir),
    )
    for line in out.splitlines():
        key, _, value = line.partition("=")
        if key == "BUCKYBALL_WORKLOAD_CORE":
            if not value:
                raise RuntimeError("workload_cmake_defs missing BUCKYBALL_WORKLOAD_CORE")
            return value
    raise RuntimeError("workload_cmake_defs missing BUCKYBALL_WORKLOAD_CORE")


def _bemu_manifest(bbdir: str, chip: str) -> str:
    defs_py = _configparse(bbdir) / "workload_cmake_defs.py"
    out = subprocess.check_output(
        [sys.executable, str(defs_py), "--repo", str(_repo(bbdir)), "--chip", chip],
        text=True,
        env=_pythonpath(bbdir),
    )
    for line in out.splitlines():
        key, _, value = line.partition("=")
        if key == "BUCKYBALL_RUSHB_BEMU_MANIFEST":
            if not value:
                raise RuntimeError("workload_cmake_defs missing BUCKYBALL_RUSHB_BEMU_MANIFEST")
            return value
    raise RuntimeError("workload_cmake_defs missing BUCKYBALL_RUSHB_BEMU_MANIFEST")


def gen_ball_isa(bbdir: str, core: str) -> Path:
    root = _repo(bbdir)
    balldomain = root / "examples" / "cores" / core / "configs" / "balldomains" / "default.toml"
    if not balldomain.is_file():
        raise RuntimeError(f"missing balldomain: {balldomain}")
    out = root / "examples" / "cores" / core / "isa" / "ballISA.h"
    out.parent.mkdir(parents=True, exist_ok=True)
    parse = _configparse(bbdir)
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        _run(
            [sys.executable, str(parse / "toml2json.py"), str(balldomain), tmp.name, "--repo", str(root)],
            bbdir=bbdir,
        )
        _run(
            [sys.executable, str(parse / "json_to_ball_isa.py"), tmp.name, str(out)],
            bbdir=bbdir,
        )
    return out


def build_workload(
    bbdir: str,
    chip: str,
    *,
    model: str = "",
    rushb: str | None = None,
    stable: bool = False,
) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", chip):
        raise ValueError(f"invalid chip: {chip}")
    if rushb is not None and rushb not in {"bemu", "verilator"}:
        raise ValueError(f"rushB must be bemu|verilator, got {rushb!r}")
    root = _repo(bbdir)
    install_bundle(bbdir, chip)
    core = _workload_core(bbdir, chip)
    build_compiler(bbdir, core=core)
    isa = gen_ball_isa(bbdir, core)
    bemu_manifest = _bemu_manifest(bbdir, chip)
    _run(
        ["cargo", "build", "--release", "--manifest-path", bemu_manifest, "--lib"],
        bbdir=bbdir,
        extra_env=bebop_cargo_env(bbdir, chip),
    )

    riscv = _require_riscv()
    linux_cc = riscv / "bin" / "riscv64-unknown-linux-gnu-gcc"
    linux_cxx = riscv / "bin" / "riscv64-unknown-linux-gnu-g++"
    if not linux_cc.is_file() or not linux_cxx.is_file():
        raise RuntimeError(f"missing RISC-V linux toolchain under {riscv / 'bin'}")

    wl = root / "bb-tests" / "workloads"
    build = Path(workload_build_dir(str(root), chip))
    defs_py = _configparse(bbdir) / "workload_cmake_defs.py"
    defs = subprocess.check_output(
        [sys.executable, str(defs_py), "--repo", str(root), "--chip", chip],
        text=True,
        env=_pythonpath(bbdir),
    )
    cmake_model, ninja_arg = _ninja_target(model.lower(), rushb)
    compiler_build = (
        root / "compiler" / "thirdparty" / "buddy-mlir" / "build" / "cores" / core
    )
    cache = compiler_build / "CMakeCache.txt"
    python_exec = subprocess.check_output(
        ["sed", "-n", r"s/^Python3_EXECUTABLE:FILEPATH=//p", str(cache)],
        text=True,
    ).splitlines()
    if not python_exec:
        raise RuntimeError(f"Python3_EXECUTABLE not found in {cache}")
    env = os.environ.copy()
    env.update(bebop_cargo_env(bbdir, chip))
    env["PATH"] = f"{riscv / 'bin'}:{env.get('PATH', '')}"
    env["RISCV"] = str(riscv)
    env["CC"] = str(linux_cc)
    env["CXX"] = str(linux_cxx)
    env["BUDDY_MLIR_BUILD_DIR"] = str(compiler_build)
    build.parent.mkdir(parents=True, exist_ok=True)
    if build.is_dir():
        import shutil

        shutil.rmtree(build)
    build.mkdir(parents=True)
    cmake_args = [
        "cmake",
        "-G",
        "Ninja",
        f"-DBUCKYBALL_STABLE={'ON' if stable else 'OFF'}",
        f"-DBB_BALL_ISA_INCLUDE_DIR={isa.parent}",
        f"-DPython3_EXECUTABLE={python_exec[0]}",
        f"-DCMAKE_C_COMPILER={linux_cc}",
        f"-DCMAKE_CXX_COMPILER={linux_cxx}",
    ]
    for line in defs.splitlines():
        if line:
            cmake_args.append(f"-D{line}")
    if cmake_model:
        cmake_args.extend(["-DMODEL=" + cmake_model, "-DARCH=buckyball"])
    cmake_args.append("..")
    result = subprocess.run(cmake_args, cwd=build, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"cmake failed ({result.returncode})")
    if ninja_arg:
        result = subprocess.run(["ninja", f"-j{os.cpu_count() or 1}", ninja_arg], cwd=build, env=env)
    else:
        result = subprocess.run(["ninja", f"-j{os.cpu_count() or 1}"], cwd=build, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"ninja failed ({result.returncode})")
