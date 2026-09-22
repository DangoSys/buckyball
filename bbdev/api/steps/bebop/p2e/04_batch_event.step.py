"""
bebop p2e batch event handler

Runs bebop p2e batch regression (aligned with runworkload):
  1. Resolve the runtime from the bitstream case
  2. Build the test harness against that runtime
  3. Run the P2E regression harness serially on one FPGA
"""
import os
import re
import shlex
import sys
from pathlib import Path

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
bebop_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if bebop_path not in sys.path:
    sys.path.insert(0, bebop_path)

from utils.event_common import require_chip
from utils.path import bebop_cargo_env, get_buckyball_path, workloads_output_root
from utils.stream_run import stream_run_logger_async
from utils.event_common import check_result, get_origin_trace_id
from utils.workload_manifest import resolve_workload_toml

config = {
    "name": "bebop-p2e-batch",
    "description": "Run bebop p2e batch regression",
    "flows": ["bebop"],
    "triggers": [queue("bebop.p2e.batch")],
    "enqueues": [],
}


def resolve_runtime_config(bitstream: str, requested_config: object) -> str:
    if isinstance(requested_config, str) and requested_config:
        return requested_config

    build_dir = os.path.dirname(os.path.dirname(os.path.abspath(bitstream)))
    case_name = os.path.basename(build_dir)
    return re.sub(r"-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}$", "", case_name)


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    bebop_dir = f"{bbdir}/bebop"
    bitstream = input_data.get("bitstream", "")
    if not bitstream or not os.path.isfile(bitstream):
        ctx.logger.error(f"bitstream .bit file not found: {bitstream}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "bitstream_not_found", "bitstream": bitstream},
            trace_id=origin_tid,
        )
        return

    bitstream = os.path.abspath(bitstream)
    build_dir = os.path.dirname(os.path.dirname(bitstream))
    if not os.path.isdir(build_dir):
        ctx.logger.error(f"P2E build case not found for bitstream: {build_dir}")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "build_dir_not_found", "build_dir": build_dir},
            trace_id=origin_tid,
        )
        return

    chip = input_data.get("chip")
    if not chip:
        ctx.logger.error("Missing required parameter: chip must be specified")
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "missing_chip"},
            trace_id=origin_tid,
        )
        return
    elf_root = workloads_output_root(bbdir)
    cargo_env = bebop_cargo_env(bbdir, chip)

    test_type = input_data.get("test", "elf-tests")
    diff = bool(input_data.get("diff", False))
    try:
        workload_toml = resolve_workload_toml(chip, "p2e", test_type, bbdir, diff=diff)
    except ValueError as e:
        ctx.logger.error(str(e))
        await check_result(
            ctx, 1, continue_run=False,
            extra_fields={"error": "invalid_regression", "test": test_type, "chip": chip},
            trace_id=origin_tid,
        )
        return

    ctx.logger.info(f"Running {test_type} with workload config: {workload_toml} diff={diff}")
    ctx.logger.info(f"P2E case dir (from bitstream): {build_dir}")

    manifest = Path(bebop_dir) / "Cargo.toml"
    features = ["p2e"]
    runtime_env = {**os.environ.copy(), **cargo_env}
    if diff:
        manifest = Path(bbdir) / "examples" / "chips" / chip / "generated" / "bebop" / "Cargo.toml"
        features.append("bemu")
        runtime_env["CARGO_TARGET_DIR"] = os.path.join(bebop_dir, "target", f"{chip}-p2e-diff")
    feature_arg = ",".join(features)
    rtcfg_path = os.path.join(build_dir, "vvacDir", "runtimeDir", "rtcfg")
    libvctb_path = os.path.join(build_dir, "vvacDir", "runtimeDir", "lib", "lib_arm", "libvCtb.so")
    runtime_artifacts = [rtcfg_path, libvctb_path, os.path.join(build_dir, "bebop-p2e")]
    if diff:
        runtime_artifacts.append(os.path.join(build_dir, "libriscv.so"))
    if not all(os.path.isfile(path) for path in runtime_artifacts):
        missing = [path for path in runtime_artifacts if not os.path.isfile(path)]
        if missing:
            ctx.logger.error(f"P2E runtime artifacts missing: {missing}")
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={
                "task": "runtime",
                "build_dir": build_dir,
                "missing": missing,
            },
            trace_id=origin_tid,
        )
        return

    # OUT_PATH makes the test harness link against this bitstream case's runtime.
    cargo_out = f"--config=\"env.OUT_PATH='{build_dir}'\""

    # ── Build bebop p2e (tests), linked against case libvCtb ───────────────
    build_cmd = (
        f"nix develop -c cargo {cargo_out} build --release "
        f"--manifest-path {shlex.quote(str(manifest))} --features {shlex.quote(feature_arg)} --tests"
    )
    ctx.logger.info("Building bebop p2e (tests)...")
    build_result = await stream_run_logger_async(
        cmd=build_cmd,
        logger=ctx.logger,
        cwd=bebop_dir,
        stdout_prefix="bebop p2e build",
        stderr_prefix="bebop p2e build",
        env=runtime_env,
    )

    if build_result.returncode != 0:
        await check_result(
            ctx, build_result.returncode, continue_run=False,
            extra_fields={"task": "build", "backend": "p2e", "build_dir": build_dir},
            trace_id=origin_tid,
        )
        return

    # ── Run regression ────────────────────────────────────────────────────
    env = runtime_env.copy()
    env["OUT_PATH"] = build_dir
    harness_args = [
        "--",
        "--workload-toml", workload_toml,
        "--bb-tests-root", elf_root,
        "--p2e-bitstream", bitstream,
    ]
    if diff:
        harness_args.append("--diff")
    harness = shlex.join(harness_args)
    test_cmd = (
        f"nix develop -c cargo {cargo_out} test --release "
        f"--manifest-path {shlex.quote(str(manifest))} --features {shlex.quote(feature_arg)} "
        f"--test test_p2e {harness}"
    )

    ctx.logger.info(f"Running bebop p2e regression: {test_cmd}")
    run_result = await stream_run_logger_async(
        cmd=test_cmd,
        logger=ctx.logger,
        cwd=bebop_dir,
        stdout_prefix="bebop p2e batch",
        stderr_prefix="bebop p2e batch",
        env=env,
    )

    extra_fields = {
        "task": "batch",
        "backend": "p2e",
        "chip": chip,
        "bitstream": bitstream,
        "build_dir": build_dir,
        "test_type": test_type,
        "diff": diff,
        "workload_toml": workload_toml,
    }

    if input_data.get("from_regression_check"):
        # Model accuracy is filled by eval-performance, not pk pass rate.
        if run_result.returncode != 0:
            await check_result(
                ctx, run_result.returncode, continue_run=False,
                extra_fields={**extra_fields, "error": "pk_tests_failed"},
                trace_id=origin_tid,
            )
            return

    await check_result(
        ctx,
        run_result.returncode,
        continue_run=False,
        extra_fields=extra_fields,
        trace_id=origin_tid,
    )
