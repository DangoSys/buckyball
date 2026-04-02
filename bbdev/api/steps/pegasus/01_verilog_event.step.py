import os
import sys

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "make pegasus verilog",
    "description": "Generate SystemVerilog from Chisel using ElaboratePegasus",
    "flows": ["pegasus"],
    "triggers": [queue("pegasus.verilog")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    arch_dir = f"{bbdir}/arch"
    build_dir = input_data.get("output_dir", f"{bbdir}/arch/build/pegasus/")

    config_name = input_data.get("config", "sims.pegasus.PegasusConfig")
    ctx.logger.info(f"[pegasus] Elaborating config: {config_name}")
    ctx.logger.info(f"[pegasus] Output directory: {build_dir}")

    os.makedirs(build_dir, exist_ok=True)

    # Use ElaboratePegasus (not the generic sims.verilator.Elaborate)
    # ElaboratePegasus instantiates PegasusHarness directly, bypassing TestHarness
    command = (
        f"mill -i __.test.runMain sims.pegasus.ElaboratePegasus "
        f"--disable-annotation-unknown "
        f"-strip-debug-info "
        f"-O=debug "
        f"--split-verilog "
        f"-o={build_dir}"
    )

    result = stream_run_logger(
        cmd=command,
        logger=ctx.logger,
        cwd=arch_dir,
        stdout_prefix="pegasus verilog",
        stderr_prefix="pegasus verilog",
    )

    # Clean up stray top-level file if emitted next to arch/
    for stray in ["PegasusHarness.sv", "TestHarness.sv"]:
        stray_path = f"{arch_dir}/{stray}"
        if os.path.exists(stray_path):
            os.remove(stray_path)

    # Copy generated SV files to pegasus/vivado/generated/ for Vivado build
    # DPI files are replaced with synth-safe stubs.
    vivado_gen_dir = f"{bbdir}/pegasus/vivado/generated"
    if result.returncode == 0 and os.path.isdir(build_dir):
        import re
        import shutil
        os.makedirs(vivado_gen_dir, exist_ok=True)
        for f in os.listdir(vivado_gen_dir):
            if f.endswith(".sv") or f.endswith(".v"):
                os.remove(os.path.join(vivado_gen_dir, f))

        def build_dpi_stub(src_path: str) -> str:
            text = open(src_path, "r", encoding="utf-8").read()
            m = re.search(r"module\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\);\s*", text, re.S)
            if m is None:
                raise RuntimeError(f"invalid DPI module format: {src_path}")
            mod_name = m.group(1)
            ports_block = m.group(2).rstrip()
            outputs = []
            for line in ports_block.splitlines():
                s = line.strip()
                if not s.startswith("output"):
                    continue
                s = s.rstrip(",")
                s = re.sub(r"^output\s+", "", s)
                s = re.sub(r"^\[[^\]]+\]\s+", "", s)
                for name in s.split(","):
                    n = name.strip()
                    if n:
                        outputs.append(n)
            stub = [f"module {mod_name}(", ports_block, ");"]
            for out_name in outputs:
                stub.append(f"  assign {out_name} = '0;")
            stub.append("endmodule")
            stub.append("")
            return "\n".join(stub)

        sv_files = [f for f in os.listdir(build_dir) if f.endswith(".sv") or f.endswith(".v")]
        for f in sv_files:
            src = os.path.join(build_dir, f)
            if "DPI" in f:
                stub_name = f"stub_{os.path.splitext(f)[0].replace('DPI', '')}.sv"
                stub_path = os.path.join(vivado_gen_dir, stub_name)
                with open(stub_path, "w", encoding="utf-8") as wf:
                    wf.write(build_dpi_stub(src))
            else:
                shutil.copy2(src, os.path.join(vivado_gen_dir, f))
        ctx.logger.info(f"[pegasus] Copied {len(sv_files)} files to {vivado_gen_dir}")

    success_result, failure_result = await check_result(
        ctx,
        result.returncode,
        continue_run=False,
        extra_fields={
            "task": "verilog",
            "output_dir": build_dir,
            "vivado_gen_dir": vivado_gen_dir,
            "top_module": "PegasusHarness",
        },
        trace_id=origin_tid,
    )

    return
