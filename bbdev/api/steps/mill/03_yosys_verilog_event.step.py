import os
import shutil
import sys
import glob
import re
import yaml

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "yosys verilog",
    "description": "generate verilog for yosys flow",
    "flows": ["yosys"],
    "triggers": [queue("yosys.run"), queue("yosys.verilog")],
    "enqueues": ["yosys.synth"],
}


def load_yosys_config():
    bbdir = get_buckyball_path()
    config_path = f"{bbdir}/bbdev/api/steps/yosys/scripts/yosys-config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def prepare_yosys_verilog(build_dir: str, yosys_log_dir: str, logger):
    def _build_stub_from_header(src_path: str, content: str, force_blackbox: bool):
        m = re.search(r"module\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*;", content, re.S)
        if m is None:
            raise RuntimeError(f"invalid module header format: {src_path}")
        mod_name = m.group(1)
        ports_block = m.group(2).rstrip()
        head = "(* blackbox *) " if force_blackbox else ""
        return f"{head}module {mod_name}(\n{ports_block}\n);\nendmodule\n"

    def _collect_defined_modules(sv_files):
        defined = set()
        for path in sv_files:
            with open(path, "r") as f:
                content = f.read()
            for m in re.finditer(r"^\s*module\s+([A-Za-z_][A-Za-z0-9_]*)\b", content, re.M):
                defined.add(m.group(1))
        return defined

    def _extract_external_modules(extern_path):
        if not os.path.exists(extern_path):
            return []
        names = []
        with open(extern_path, "r") as f:
            for line in f:
                m = re.match(r"\s*//\s*external module\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", line)
                if m:
                    names.append(m.group(1))
        return names

    def _collect_ports(mod_name, sv_files):
        ports = set()
        inst_re = re.compile(rf"\b{mod_name}\b\s+[A-Za-z_][A-Za-z0-9_]*\s*\((.*?)\)\s*;", re.S)
        port_re = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")
        for path in sv_files:
            with open(path, "r") as f:
                content = f.read()
            for im in inst_re.finditer(content):
                block = im.group(1)
                for pm in port_re.finditer(block):
                    ports.add(pm.group(1))
        return sorted(ports)

    def _build_blackbox(mod_name, ports):
        lines = [f"(* blackbox *) module {mod_name}("]
        if ports:
            for idx, p in enumerate(ports):
                end = "," if idx < len(ports) - 1 else ""
                lines.append(f"  input {p}{end}")
        lines.append(");")
        lines.append("endmodule")
        lines.append("")
        return "\n".join(lines)

    stub_dir = os.path.join(yosys_log_dir, "dpi_stubs")
    os.makedirs(stub_dir, exist_ok=True)

    all_sv = glob.glob(f"{build_dir}/**/*.sv", recursive=True)
    kept = []
    dpi_stubbed = []
    pattern_stubbed = []
    skipped = []

    for path in all_sv:
        with open(path, "r") as f:
            content = f.read()
        if 'import "DPI-C"' in content:
            stub_path = os.path.join(stub_dir, f"stub_{os.path.basename(path)}")
            with open(stub_path, "w") as wf:
                wf.write(_build_stub_from_header(path, content, False))
            kept.append(stub_path)
            dpi_stubbed.append(os.path.basename(path))
        elif "'{" in content:
            stub_path = os.path.join(stub_dir, f"pattern_stub_{os.path.basename(path)}")
            try:
                with open(stub_path, "w") as wf:
                    wf.write(_build_stub_from_header(path, content, True))
                kept.append(stub_path)
                pattern_stubbed.append(os.path.basename(path))
            except RuntimeError:
                skipped.append(os.path.basename(path))
        else:
            kept.append(path)

    extern_path = os.path.join(build_dir, "extern_modules.sv")
    extern_names = _extract_external_modules(extern_path)
    defined = _collect_defined_modules(kept)
    ext_stubbed = []
    for mod_name in extern_names:
        if mod_name in defined:
            continue
        ports = _collect_ports(mod_name, all_sv)
        stub_path = os.path.join(stub_dir, f"ext_stub_{mod_name}.sv")
        with open(stub_path, "w") as f:
            f.write(_build_blackbox(mod_name, ports))
        kept.append(stub_path)
        ext_stubbed.append(mod_name)

    if not kept:
        raise RuntimeError("no yosys source generated")

    source_list_path = os.path.join(build_dir, "yosys_sources.list")
    with open(source_list_path, "w") as f:
        for path in kept:
            f.write(path + "\n")

    if dpi_stubbed:
        logger.info(
            f"Stubbed {len(dpi_stubbed)} DPI modules: {', '.join(dpi_stubbed[:10])}{'...' if len(dpi_stubbed) > 10 else ''}"
        )
    if pattern_stubbed:
        logger.info(
            f"Stubbed {len(pattern_stubbed)} pattern modules: {', '.join(pattern_stubbed[:10])}{'...' if len(pattern_stubbed) > 10 else ''}"
        )
    if ext_stubbed:
        logger.info(
            f"Stubbed {len(ext_stubbed)} external modules: {', '.join(ext_stubbed[:10])}{'...' if len(ext_stubbed) > 10 else ''}"
        )
    if skipped:
        logger.info(
            f"Skipped {len(skipped)} unsupported files: {', '.join(skipped[:10])}{'...' if len(skipped) > 10 else ''}"
        )
    logger.info(f"Prepared yosys source list: {source_list_path}")
    return source_list_path


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    build_dir = input_data.get("output_dir", f"{bbdir}/arch/build/")
    arch_dir = f"{bbdir}/arch"

    yosys_cfg = load_yosys_config()
    elaborate_config = input_data.get("config") or yosys_cfg.get(
        "elaborate_config", "sims.verilator.BuckyballToyVerilatorConfig"
    )

    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    verilog_command = (
        f"mill -i __.test.runMain sims.verilator.Elaborate {elaborate_config} "
        "--disable-annotation-unknown -strip-debug-info -O=debug "
        "-lowering-options=disallowLocalVariables "
        f"--split-verilog -o={build_dir}"
    )

    result = stream_run_logger(
        cmd=verilog_command,
        logger=ctx.logger,
        cwd=arch_dir,
        stdout_prefix="yosys verilog",
        stderr_prefix="yosys verilog",
    )

    if result.returncode != 0:
        success_result, failure_result = await check_result(
            ctx,
            result.returncode,
            continue_run=False,
            extra_fields={"task": "verilog"},
            trace_id=origin_tid,
        )
        return failure_result

    for unwanted in ["TestHarness.sv", "TargetBall.sv"]:
        topname_file = f"{arch_dir}/{unwanted}"
        if os.path.exists(topname_file):
            os.remove(topname_file)

    yosys_log_dir = f"{bbdir}/bbdev/api/steps/yosys/log"
    os.makedirs(yosys_log_dir, exist_ok=True)
    try:
        source_list_path = prepare_yosys_verilog(build_dir, yosys_log_dir, ctx.logger)
    except Exception as e:
        success_result, failure_result = await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "verilog", "error": str(e)},
            trace_id=origin_tid,
        )
        return failure_result

    await check_result(
        ctx,
        result.returncode,
        continue_run=input_data.get("from_run_workflow", False),
        extra_fields={"task": "verilog", "source_list": source_list_path},
        trace_id=origin_tid,
    )

    if input_data.get("from_run_workflow"):
        await ctx.enqueue({"topic": "yosys.synth", "data": {**input_data, "task": "run"}})

    return
