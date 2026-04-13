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
    "description": "Generate SystemVerilog from Chisel using ElaboratePegasusTop",
    "flows": ["pegasus"],
    "triggers": [queue("pegasus.verilog")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    arch_dir = f"{bbdir}/arch"
    build_dir = input_data.get("output_dir", f"{bbdir}/arch/build/pegasus/")

    build_dir = input_data.get("output_dir", f"{bbdir}/arch/build/pegasus_top/")
    soc_dir   = f"{bbdir}/arch/build/pegasus/"   # DigitalTop + all SoC RTL

    ctx.logger.info(f"[pegasus] Elaborating PegasusTop")
    ctx.logger.info(f"[pegasus] Top output directory: {build_dir}")
    ctx.logger.info(f"[pegasus] SoC RTL directory: {soc_dir}")

    os.makedirs(build_dir, exist_ok=True)

    # Step 1: Elaborate the full SoC (DigitalTop + all sub-modules)
    # This is the same as before, using ElaboratePegasus which runs PegasusHarness
    # but we only need its generated RTL files (DigitalTop.sv, BBTile.sv, etc.)
    soc_command = (
        f"mill -i __.test.runMain sims.pegasus.ElaboratePegasus "
        f"--disable-annotation-unknown "
        f"-strip-debug-info "
        f"--split-verilog "
        f"-o={soc_dir}"
    )
    soc_result = stream_run_logger(
        cmd=soc_command,
        logger=ctx.logger,
        cwd=arch_dir,
        stdout_prefix="pegasus soc verilog",
        stderr_prefix="pegasus soc verilog",
    )
    if soc_result.returncode != 0:
        ctx.logger.error("[pegasus] SoC elaboration failed")
        await check_result(ctx, soc_result.returncode, continue_run=False,
                           extra_fields={"task": "verilog", "step": "soc"}, trace_id=origin_tid)
        return

    # Step 2: Elaborate PegasusTop + PegasusShell (the FPGA top-level wrapper)
    command = (
        f"mill -i __.test.runMain sims.pegasus.ElaboratePegasusTop "
        f"--disable-annotation-unknown "
        f"-strip-debug-info "
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

    # Clean up stray top-level files if emitted next to arch/
    for stray in ["PegasusTop.v", "PegasusTopWrapper.sv", "PegasusShell.v"]:
        stray_path = f"{arch_dir}/{stray}"
        if os.path.exists(stray_path):
            os.remove(stray_path)

    # Copy generated SV/V files to pegasus/vivado/generated/ for Vivado build.
    # Sources:
    #   soc_dir   -> DigitalTop + all SoC sub-modules (DPI files stubbed out)
    #   build_dir -> PegasusTop.v, PegasusShell.v, PegasusTopWrapper.sv (top-level wrappers)
    # PegasusHarness.sv, ChipTop.sv and harness-layer files from soc_dir are skipped.
    vivado_gen_dir = f"{bbdir}/thirdparty/pegasus/vivado/generated"
    if result.returncode == 0 and os.path.isdir(build_dir):
        import re
        import shutil
        os.makedirs(vivado_gen_dir, exist_ok=True)
        for f in os.listdir(vivado_gen_dir):
            if f.endswith(".sv") or f.endswith(".v"):
                os.remove(os.path.join(vivado_gen_dir, f))

        HARNESS_SKIP = {"PegasusHarness.sv", "ChipTop.sv"}

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

        def copy_rtl_dir(src_dir: str, skip_set: set = None) -> int:
            copied = 0
            for f in os.listdir(src_dir):
                if not (f.endswith(".sv") or f.endswith(".v")):
                    continue
                if skip_set and f in skip_set:
                    continue
                src = os.path.join(src_dir, f)
                if "DPI" in f:
                    stub_name = f"stub_{os.path.splitext(f)[0].replace('DPI', '')}.sv"
                    with open(os.path.join(vivado_gen_dir, stub_name), "w") as wf:
                        wf.write(build_dpi_stub(src))
                else:
                    shutil.copy2(src, os.path.join(vivado_gen_dir, f))
                copied += 1
            return copied

        # Copy SoC RTL (skip harness-layer files)
        n_soc = copy_rtl_dir(soc_dir, skip_set=HARNESS_SKIP)
        # Copy top-level wrappers (overrides any same-named file from soc_dir)
        n_top = copy_rtl_dir(build_dir)
        ctx.logger.info(f"[pegasus] Copied {n_soc} SoC files + {n_top} top files to {vivado_gen_dir}")

    success_result, failure_result = await check_result(
        ctx,
        result.returncode,
        continue_run=False,
        extra_fields={
            "task": "verilog",
            "output_dir": build_dir,
            "vivado_gen_dir": vivado_gen_dir,
            "top_module": "PegasusTop",
        },
        trace_id=origin_tid,
    )

    return
