"""
bebop bemu event handler

Run BEMU either through its guest-ELF emulator or the rushB native ABI.
"""
import os
import shlex
import sys
import tomllib
from datetime import datetime
from pathlib import Path

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from utils.path import bebop_cargo_env, bebop_target_dir, get_buckyball_path, get_bemu_log_dir, workload_tests_root
from utils.stream_run import stream_run_logger_async
from utils.search_workload import search_workload, search_workload_all
from utils.event_common import check_result, get_origin_trace_id
from utils.process_registry import cancellation_requested
from bemu_common import bemu_core_manifest, bemu_manifest, bemu_tile_index, chip_emu_manifest

PERFETTO_TARGETS = {
    "buddy-buckyball-lenet-run": "buddy-buckyball-lenet-perfetto",
}

config = {
    "name": "bebop-bemu-sim",
    "description": "Run bebop bemu emulator",
    "flows": ["bebop"],
    "triggers": [queue("bebop.bemu.sim")],
    "enqueues": [],
}


def clean_model_trace(binary_dir: str) -> None:
    trace_dir = Path(binary_dir) / "trace"
    for subdir in ("cycle", "tensor"):
        target_dir = trace_dir / subdir
        if not target_dir.exists():
            continue
        if not target_dir.is_dir():
            raise NotADirectoryError(f"trace path is not a directory: {target_dir}")
        for path in target_dir.glob("trace-*.txt"):
            if not path.is_file():
                raise FileNotFoundError(f"trace path is not a file: {path}")
            path.unlink()
        summary = target_dir / "summary.txt"
        if summary.exists():
            if not summary.is_file():
                raise FileNotFoundError(f"trace summary path is not a file: {summary}")
            summary.unlink()

    perfetto = trace_dir / "perfetto.json"
    if perfetto.exists():
        if not perfetto.is_file():
            raise FileNotFoundError(f"perfetto path is not a file: {perfetto}")
        perfetto.unlink()


def resolve_bemu_binary(bbdir: str, chip: str, binary_name: str) -> str | None:
    if Path(binary_name).name != binary_name:
        return None

    workload_root = workload_tests_root(bbdir, chip)
    chip_root = f"{workload_root}/CTest/chips/{chip}"
    chip_binary = search_workload(chip_root, binary_name)
    if chip_binary is not None:
        return chip_binary

    matches = search_workload_all(workload_root, binary_name)
    chip_marker = f"{os.path.sep}CTest{os.path.sep}chips{os.path.sep}"
    non_chip_matches = [path for path in matches if chip_marker not in path]
    if len(non_chip_matches) == 1:
        return non_chip_matches[0]
    return None


def is_native_host_elf(path: str) -> bool:
    try:
        with open(path, "rb") as binary:
            header = binary.read(20)
    except OSError:
        return False
    return header[:4] == b"\x7fELF" and header[18:20] == b"\x3e\x00"


def rushb_bemu_library(manifest: Path, bbdir: str, chip: str) -> Path:
    """Return the cdylib emitted by bebop-bemu for a rushB run."""
    with manifest.open("rb") as source:
        cargo = tomllib.load(source)
    lib_name = cargo.get("lib", {}).get("name")
    if not lib_name:
        lib_name = cargo["package"]["name"].replace("-", "_")
    return Path(bebop_target_dir(bbdir, chip)) / "release" / f"lib{lib_name}.so"


def rushb_bemu_library_dirs(manifest: Path, bbdir: str, chip: str) -> list[Path]:
    """Locate native shared-library dependencies produced by the bebop-bemu build."""
    build_dir = Path(bebop_target_dir(bbdir, chip)) / "release" / "build"
    return sorted({library.parent for library in build_dir.glob("*/out/**/libriscv.so")})


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()

    chip = input_data.get("chip")
    if not chip:
        ctx.logger.error("Missing required parameter: chip must be specified")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "missing_chip"},
            trace_id=origin_tid,
        )
        return
    try:
        bemu_cargo_manifest = bemu_manifest(chip, bbdir)
    except ValueError as e:
        ctx.logger.error(str(e))
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "invalid_chip", "chip": chip},
            trace_id=origin_tid,
        )
        return

    binary_name = input_data.get("binary", "")
    binary_path = resolve_bemu_binary(bbdir, chip, binary_name)
    if binary_path is None:
        ctx.logger.error(f"binary not found: {binary_name}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "binary_not_found", "binary": binary_name},
            trace_id=origin_tid,
        )
        return
    ctx.logger.info(f"binary_path: {binary_path}")
    binary_dir = os.path.dirname(binary_path)
    perfetto_target = PERFETTO_TARGETS.get(binary_name)
    if perfetto_target:
        clean_model_trace(binary_dir)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_dir = get_bemu_log_dir(bbdir, chip, timestamp, binary_name)
    os.makedirs(log_dir, exist_ok=True)

    if input_data.get("rushB"):
        # rushB is a native ABI for one accelerator instance, so it remains a
        # Core-level backend even when guest-ELF BEMU runs a whole tile.
        bemu_cargo_manifest = bemu_core_manifest(chip, bbdir)
        # rushB binaries are host executables. Rebuilding and co-locating the
        # backend library keeps the ABI selection explicit and avoids sending
        # a host ELF through Spike's guest-ELF path.
        if not is_native_host_elf(binary_path):
            ctx.logger.error(
                f"--rushB requires a native x86_64 rushB runner; got guest ELF: {binary_path}"
            )
            await check_result(
                ctx, 1, continue_run=False,
                extra_fields={"error": "rushB_requires_native_runner", "binary": binary_path},
                trace_id=origin_tid,
            )
            return
        bemu_library = rushb_bemu_library(bemu_cargo_manifest, bbdir, chip)
        bemu_runtime_library = Path(binary_dir) / "libbebop_bemu.so"
        dependency_dirs = rushb_bemu_library_dirs(bemu_cargo_manifest, bbdir, chip)
        build_cmd = shlex.join([
            "cargo", "build", "--release", "--manifest-path", str(bemu_cargo_manifest), "--lib",
        ])
        copy_cmd = shlex.join([
            "cmake", "-E", "copy_if_different", str(bemu_library), str(bemu_runtime_library),
        ])
        dependency_path = os.pathsep.join(str(path) for path in dependency_dirs)
        library_env = f"LD_LIBRARY_PATH={shlex.quote(dependency_path)}:${{LD_LIBRARY_PATH:-}}"
        inner_cmd = f"cd {shlex.quote(binary_dir)} && {build_cmd} && {copy_cmd} && {library_env} exec {shlex.quote(binary_path)}"
        run_cmd = f"nix develop -c sh -c {shlex.quote(inner_cmd)}"
        ctx.logger.info(f"Running rushB BEMU: {run_cmd}")
        run_result = await stream_run_logger_async(
            cmd=run_cmd,
            logger=ctx.logger,
            cwd=bbdir,
            stdout_prefix="rushB bemu",
            stderr_prefix="rushB bemu",
            task_scope=origin_tid,
            env={**os.environ.copy(), **bebop_cargo_env(bbdir, chip)},
        )
        if cancellation_requested(origin_tid):
            return
        await check_result(
            ctx,
            run_result.returncode,
            continue_run=False,
            extra_fields={
                "task": "bemu",
                "backend": "rushB",
                "binary": binary_path,
                "chip": chip,
                "log_dir": log_dir,
                "timestamp": timestamp,
            },
            trace_id=origin_tid,
        )
        return

    # ── Run bebop bemu ────────────────────────────────────────────────────
    tile_index = bemu_tile_index(chip, bbdir)
    chip_emu = chip_emu_manifest(chip, bbdir)
    if (tile_index is not None) != bool(chip_emu):
        ctx.logger.error(
            f"chip {chip}: bundle bemu.chipMain and emu/Cargo.toml must both exist or both be absent"
        )
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "chip_emu_entry_mismatch", "chip": chip},
            trace_id=origin_tid,
        )
        return
    if chip_emu:
        cargo_args = [
            "cargo",
            "run",
            "--release",
            "--manifest-path",
            str(chip_emu),
            "--",
            "--tile-index",
            str(tile_index),
            "--elf",
            binary_path,
            "--log-dir",
            log_dir,
        ]
    else:
        cargo_args = [
            "cargo",
            "run",
            "--release",
            "--manifest-path",
            str(bemu_cargo_manifest),
            "--bin",
            "bebop-bemu",
            "--",
            "--elf",
            binary_path,
            "--log-dir",
            log_dir,
        ]
    if input_data.get("pk"):
        cargo_args.append("--pk")
    if input_data.get("disasm"):
        cargo_args.append("--disasm")
    if input_data.get("tool-profile"):
        cargo_args.append("--tool-profile")
    for trace_name in ("itrace", "mtrace"):
        if input_data.get(trace_name, False):
            cargo_args.append(f"--{trace_name}")
    inner_cmd = f"cd {shlex.quote(binary_dir)} && {shlex.join(cargo_args)}"
    run_cmd = f"nix develop -c sh -c {shlex.quote(inner_cmd)}"
    ctx.logger.info(f"Running bebop bemu: {run_cmd}")
    run_result = await stream_run_logger_async(
        cmd=run_cmd,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="bebop bemu",
        stderr_prefix="bebop bemu",
        task_scope=origin_tid,
        env={**os.environ.copy(), **bebop_cargo_env(bbdir, chip)},
    )
    if cancellation_requested(origin_tid):
        return
    if run_result.returncode != 0:
        await check_result(
            ctx,
            run_result.returncode,
            continue_run=False,
            extra_fields={
                "task": "bemu",
                "binary": binary_path,
                "chip": chip,
                "log_dir": log_dir,
                "timestamp": timestamp,
            },
            trace_id=origin_tid,
        )
        return

    perfetto_path = None
    if perfetto_target:
        perfetto_cmd = (
            f"cmake --build {shlex.quote(f'{bbdir}/bb-tests/build')} "
            f"--target {shlex.quote(perfetto_target)}"
        )
        ctx.logger.info(f"Generating Perfetto trace: {perfetto_cmd}")
        perfetto_result = await stream_run_logger_async(
            cmd=perfetto_cmd,
            logger=ctx.logger,
            cwd=bbdir,
            stdout_prefix="perfetto",
            stderr_prefix="perfetto",
        )
        perfetto_path = f"{binary_dir}/trace/perfetto.json"
        if perfetto_result.returncode != 0:
            await check_result(
                ctx,
                perfetto_result.returncode,
                continue_run=False,
                extra_fields={
                    "task": "perfetto",
                    "binary": binary_path,
                    "log_dir": log_dir,
                    "timestamp": timestamp,
                    "perfetto_target": perfetto_target,
                    "perfetto": perfetto_path,
                },
                trace_id=origin_tid,
            )
            return

    await check_result(
        ctx,
        0,
        continue_run=False,
        extra_fields={
            "task": "bemu",
            "binary": binary_path,
            "chip": chip,
            "log_dir": log_dir,
            "timestamp": timestamp,
            "perfetto_target": perfetto_target,
            "perfetto": perfetto_path,
        },
        trace_id=origin_tid,
    )
