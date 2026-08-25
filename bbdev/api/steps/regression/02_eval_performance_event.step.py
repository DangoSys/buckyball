"""regression --eval-performance event.

Per model (synchronous shells inside one regression event):
  1. workload build --chip <chip> --model <m>
  2. bridge chip layout output -> flat path the kernel expects
  3. kernel build --model <m> with dataset packed from e2e/datasets/
     -> fw_payload-<m>.hex; /init runs binary with --dataset/--max-samples
  4. p2e runworkload; parse per-model accuracy from uart (top1= / map=)
  5. require non-empty log_dir/trace/cycle/; run perfetto.py; cycles_i

After all models: cycles = mean(cycles_i); accuracy = mean(accuracy_i);
merge_metrics(cycles=..., accuracy=..., models=[...]).
Any model failure aborts the whole stage (no partial average).
"""
import importlib.util
import json
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path, rtl_dir, log_dir
from utils.stream_run import stream_run_logger_async
from utils.event_common import check_result, get_origin_trace_id

sys.path.insert(0, os.path.join(get_buckyball_path(), "bb-tests", "workloads", "scripts"))
import build as workload_build  # noqa: E402

regression_scripts = os.path.join(os.path.dirname(__file__), "scripts")
if regression_scripts not in sys.path:
    sys.path.insert(0, regression_scripts)
from model_layout import bridge_model_layout, perfetto_inputs
from models_toml import load_eval_models
from accuracy import accuracy_from_uart, load_eval_accuracy, mean_accuracy
from perfetto_latency import e2e_cycles_from_perfetto, mean_cycles
from result import merge_metrics

config = {
    "name": "regression-eval-performance",
    "description": "Per-model p2e perfetto cycles for regression",
    "flows": ["regression"],
    "triggers": [queue("regression.eval-performance")],
    "enqueues": [],
}


def _load_sibling(rel):
    path = os.path.normpath(os.path.join(os.path.dirname(__file__), rel))
    name = os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load sibling step: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_workload = _load_sibling("../workload/01_build_event.step.py")
_kernel = _load_sibling("../kernel/01_build_event.step.py")
_p2e = _load_sibling("../bebop/p2e/03_runworkload_event.step.py")

MODEL_LAYOUT = _workload.MODEL_LAYOUT
PERFETTO_SCRIPT = (
    Path(get_buckyball_path())
    / "bb-tests" / "workloads" / "src" / "ModelTest"
    / "e2e" / "framework" / "trace" / "perfetto.py"
)


def _fail(ctx, origin_tid, error, **extra):
    fields = {"task": "regression.eval-performance", "error": error}
    fields.update(extra)
    return check_result(ctx, 1, continue_run=False, extra_fields=fields, trace_id=origin_tid)


def _workload_build(bbdir, chip, model):
    model_key = model.lower()
    if MODEL_LAYOUT.get(model_key) is None:
        raise ValueError(f"Unknown model: {model}")
    workload_build.build_workload(bbdir, chip, model=model_key)


def _kernel_build_cmds(bbdir, model, dataset=""):
    kernel_src = os.path.join(bbdir, "bb-tests", "workloads", "lib", "kernel")
    hart_params = {"visible": 64, "total": 64, "hidden_base": 64}
    kernel_build = _kernel.kernel_build_dir(bbdir, hart_params, model=model)
    ds_arg = ""
    if dataset:
        ds_arg = f" -DBUCKYBALL_MODEL_DATASET={shlex.quote(dataset)}"
    configure = (
        f"cmake -B {kernel_build} -S {kernel_src} "
        f"-DBUCKYBALL_VISIBLE_HART_COUNT=64 "
        f"-DBUCKYBALL_TOTAL_HART_COUNT=64 "
        f"-DBUCKYBALL_HIDDEN_HART_BASE=64 "
        f"-DBUCKYBALL_KERNEL_MODEL={model} "
        f"-DBUCKYBALL_KERNEL_CHIP= "
        f"-DBUCKYBALL_KERNEL_INTERACTIVE=OFF"
        f"{ds_arg}"
    )
    build = f"cmake --build {kernel_build} --target kernel-build"
    payload = _kernel.fw_payload_name(hart_params, model=model)
    fw_bin = os.path.join(bbdir, "bb-tests", "output", "kernel", f"{payload}.bin")
    fw_hex = os.path.join(bbdir, "bb-tests", "output", "kernel", f"{payload}.hex")
    return configure, build, fw_bin, fw_hex


def _p2e_run_cmds(bbdir, bitstream, image_name, chip, input_data):
    image_path = _p2e.resolve_image(bbdir, image_name, chip)
    if not image_path:
        raise FileNotFoundError(
            f"image .hex not found for name: {image_name} "
            f"(searched bb-tests/output/{chip}/workloads/)"
        )
    bitstream = os.path.abspath(bitstream)
    build_dir = os.path.dirname(os.path.dirname(bitstream))
    if not os.path.isdir(build_dir):
        raise FileNotFoundError(
            f"P2E build case not found for bitstream: {build_dir}"
        )
    multi_fpga = bool(input_data.get("multi-fpga", False))
    if not multi_fpga and _p2e.case_uses_multi_fpga(build_dir):
        multi_fpga = True
    vsrc_dir = rtl_dir(bbdir, chip, "p2e", input_data.get("vsrc_dir"))
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_log = log_dir(bbdir, chip, "p2e", timestamp, "p2e", image_name, input_data.get("vsrc_dir"))
    os.makedirs(run_log, exist_ok=True)
    runtime_cmd = (
        f"env BEBOP_P2E_RUNTIME_ONLY=1 BEBOP_P2E_REBUILD_RUNTIME=1 "
        f'cargo run --release --features p2e -- build p2e '
        f'--rtl-dir="{vsrc_dir}" '
        f'--out-dir="{build_dir}"'
    )
    run_cmd = (
        f"cargo run --release --features p2e "
        f"--config=\"env.OUT_PATH='{build_dir}'\" "
        f"-- run p2e "
        f'--image="{image_path}" '
        f'--bitstream="{bitstream}" '
        f'--log-dir="{log_dir}"'
    )
    if multi_fpga:
        run_cmd += " --multi-fpga"
    return runtime_cmd, run_cmd, f"{bbdir}/bebop", run_log, build_dir


def _perfetto_cmd(trace_dir, trace_toml, mlir_files):
    args = [str(trace_dir), str(trace_toml)]
    for mlir in mlir_files:
        args += ["--mlir", str(mlir)]
    return (
        f"{sys.executable} {shlex.quote(str(PERFETTO_SCRIPT))} "
        + " ".join(shlex.quote(a) for a in args)
    )


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()

    chip = input_data.get("chip")
    if not isinstance(chip, str) or not chip or chip == "None":
        ctx.logger.error("Missing required parameter: chip")
        await _fail(ctx, origin_tid, "missing_chip")
        return
    bitstream = input_data.get("bitstream")
    if not isinstance(bitstream, str) or not bitstream or bitstream == "None":
        ctx.logger.error("Missing required parameter: bitstream")
        await _fail(ctx, origin_tid, "missing_bitstream")
        return
    if not os.path.isfile(bitstream):
        ctx.logger.error(f"bitstream file not found: {bitstream}")
        await _fail(ctx, origin_tid, "bitstream_not_found", bitstream=bitstream)
        return

    try:
        models = load_eval_models(chip, bbdir)
        acc_specs = load_eval_accuracy(chip, bbdir, models)
    except ValueError as e:
        ctx.logger.error(str(e))
        await _fail(ctx, origin_tid, "models_toml_invalid", chip=chip)
        return

    await check_result(
        ctx, 0, continue_run=True,
        extra_fields={
            "task": "regression.eval-performance",
            "chip": chip, "bitstream": bitstream, "models": models,
        },
        trace_id=origin_tid,
    )

    model_results = []
    for model in models:
        model_key = model.lower()
        if MODEL_LAYOUT.get(model_key) is None:
            ctx.logger.error(f"Unknown model: {model}")
            await _fail(ctx, origin_tid, "unknown_model", model=model)
            return

        ctx.logger.info(f"[eval-performance] model {model}: workload build")
        try:
            _workload_build(bbdir, chip, model)
        except (ValueError, RuntimeError) as e:
            ctx.logger.error(str(e))
            await _fail(ctx, origin_tid, "workload_build_cmd", model=model)
            return

        ctx.logger.info(f"[eval-performance] model {model}: bridge layout")
        try:
            bridge_model_layout(bbdir, chip, model)
        except (FileNotFoundError, FileExistsError) as e:
            ctx.logger.error(str(e))
            await _fail(ctx, origin_tid, "layout_bridge_failed", model=model)
            return

        ctx.logger.info(f"[eval-performance] model {model}: kernel build")
        spec = acc_specs[model]
        try:
            k_cfg, k_build, fw_bin, fw_hex = _kernel_build_cmds(
                bbdir, model, dataset=spec["dataset"],
            )
        except ValueError as e:
            ctx.logger.error(str(e))
            await _fail(ctx, origin_tid, "kernel_build_cmd", model=model)
            return
        cfg_result = await stream_run_logger_async(
            cmd=k_cfg, logger=ctx.logger,
            stdout_prefix=f"kernel configure {model}",
            stderr_prefix=f"kernel configure {model}",
        )
        if cfg_result.returncode != 0:
            await _fail(ctx, origin_tid, "kernel_configure_failed",
                        model=model, returncode=cfg_result.returncode)
            return
        build_result = await stream_run_logger_async(
            cmd=k_build, logger=ctx.logger,
            stdout_prefix=f"kernel build {model}",
            stderr_prefix=f"kernel build {model}",
        )
        if build_result.returncode != 0:
            await _fail(ctx, origin_tid, "kernel_build_failed",
                        model=model, returncode=build_result.returncode)
            return
        if not os.path.isfile(fw_bin):
            ctx.logger.error(f"fw_payload bin not found: {fw_bin}")
            await _fail(ctx, origin_tid, "kernel_bin_missing",
                        model=model, fw_bin=fw_bin)
            return
        if not _kernel.bin_to_hex(fw_bin, fw_hex, base_address=0x80000000):
            ctx.logger.error(f"bin_to_hex failed: {fw_bin} -> {fw_hex}")
            await _fail(ctx, origin_tid, "kernel_tohex_failed", model=model)
            return

        ctx.logger.info(f"[eval-performance] model {model}: p2e runworkload")
        image_name = os.path.splitext(os.path.basename(fw_hex))[0]
        try:
            runtime_cmd, run_cmd, bebop_cwd, run_log, build_dir = _p2e_run_cmds(
                bbdir, bitstream, image_name, chip, input_data
            )
        except FileNotFoundError as e:
            ctx.logger.error(str(e))
            err = "p2e_run_cmd"
            extra = {"model": model}
            if str(e).startswith("P2E build case not found"):
                err = "build_dir_not_found"
                extra["build_dir"] = os.path.dirname(
                    os.path.dirname(os.path.abspath(bitstream))
                )
            await _fail(ctx, origin_tid, err, **extra)
            return
        rt_result = await stream_run_logger_async(
            cmd=runtime_cmd, logger=ctx.logger, cwd=bebop_cwd,
            stdout_prefix=f"p2e runtime {model}",
            stderr_prefix=f"p2e runtime {model}",
        )
        rtcfg = os.path.join(build_dir, "vvacDir", "runtimeDir", "rtcfg")
        libvctb = os.path.join(
            build_dir, "vvacDir", "runtimeDir", "lib", "lib_arm", "libvCtb.so"
        )
        if rt_result.returncode != 0 or not all(
            os.path.isfile(p) for p in (rtcfg, libvctb)
        ):
            await _fail(ctx, origin_tid, "p2e_runtime_failed",
                        model=model, returncode=rt_result.returncode)
            return
        run_result = await stream_run_logger_async(
            cmd=run_cmd, logger=ctx.logger, cwd=bebop_cwd,
            stdout_prefix=f"p2e run {model}",
            stderr_prefix=f"p2e run {model}",
        )
        if run_result.returncode != 0:
            await _fail(ctx, origin_tid, "p2e_run_failed",
                        model=model, returncode=run_result.returncode,
                        log_dir=log_dir)
            return

        uart_path = Path(run_log) / "uart_hart_0.log"
        if not uart_path.is_file():
            uart_path = Path(run_log) / "uart.log"
        if not uart_path.is_file():
            ctx.logger.error(f"uart log missing under {log_dir}")
            await _fail(ctx, origin_tid, "uart_missing", model=model, log_dir=log_dir)
            return
        try:
            acc_i = accuracy_from_uart(spec["metric"], uart_path.read_text())
        except ValueError as e:
            ctx.logger.error(f"accuracy parse failed for {model}: {e}")
            await _fail(ctx, origin_tid, "accuracy_unparsable", model=model)
            return

        cycle_dir = Path(run_log) / "trace" / "cycle"
        if not cycle_dir.is_dir():
            ctx.logger.error(f"cycle trace dir missing: {cycle_dir}")
            await _fail(ctx, origin_tid, "cycle_trace_missing",
                        model=model, cycle_dir=str(cycle_dir))
            return
        cycle_files = [
            p for p in cycle_dir.iterdir()
            if p.is_file() and p.name.startswith("trace-")
            and p.name.endswith(".txt")
        ]
        if not cycle_files:
            ctx.logger.error(f"cycle trace dir empty: {cycle_dir}")
            await _fail(ctx, origin_tid, "cycle_trace_empty",
                        model=model, cycle_dir=str(cycle_dir))
            return

        ctx.logger.info(f"[eval-performance] model {model}: perfetto")
        try:
            pinputs = perfetto_inputs(bbdir, chip, model)
        except (KeyError, FileNotFoundError) as e:
            ctx.logger.error(str(e))
            await _fail(ctx, origin_tid, "perfetto_inputs_missing", model=model)
            return
        trace_dir = Path(run_log) / "trace"
        perf_cmd = _perfetto_cmd(
            trace_dir, pinputs["trace_toml"], pinputs["mlir_files"]
        )
        perf_out = trace_dir / "perfetto.json"
        perf_result = await stream_run_logger_async(
            cmd=perf_cmd, logger=ctx.logger,
            stdout_prefix=f"perfetto {model}",
            stderr_prefix=f"perfetto {model}",
        )
        if perf_result.returncode != 0 or not perf_out.is_file():
            await _fail(ctx, origin_tid, "perfetto_failed",
                        model=model, returncode=perf_result.returncode)
            return
        try:
            perf_data = json.loads(perf_out.read_text())
        except (OSError, json.JSONDecodeError) as e:
            ctx.logger.error(f"perfetto json unreadable: {e}")
            await _fail(ctx, origin_tid, "perfetto_json_unreadable", model=model)
            return
        try:
            cycles_i = e2e_cycles_from_perfetto(perf_data)
        except ValueError as e:
            ctx.logger.error(str(e))
            await _fail(ctx, origin_tid, "perfetto_cycles_failed", model=model)
            return

        dest = Path(run_log) / f"perfetto-{model}.json"
        dest.write_text(json.dumps(perf_data, indent=2))
        model_results.append({
            "name": model,
            "cycles": cycles_i,
            "accuracy": acc_i,
            "metric": spec["metric"],
            "dataset": spec["dataset"],
            "perfetto": str(dest),
        })

    try:
        cycles = mean_cycles([r["cycles"] for r in model_results])
        accuracy = mean_accuracy([r["accuracy"] for r in model_results])
    except ValueError as e:
        ctx.logger.error(str(e))
        await _fail(ctx, origin_tid, "cycles_mean_failed", models=model_results)
        return
    merge_metrics(bbdir, cycles=cycles, accuracy=accuracy, models=model_results)
    await check_result(
        ctx, 0, continue_run=False,
        extra_fields={
            "task": "regression.eval-performance",
            "chip": chip, "cycles": cycles, "accuracy": accuracy,
            "models": model_results,
        },
        trace_id=origin_tid,
    )
