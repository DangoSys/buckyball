import glob
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.event_common import check_result, get_origin_trace_id
from utils.event_common import require_chip
from utils.path import get_buckyball_path, rtl_dir
from utils.stream_run import stream_run_logger_async


config = {
    "name": "vcs-build",
    "description": "compile BBSimHarness with Synopsys VCS",
    "flows": ["vcs"],
    "triggers": [queue("vcs.build")],
    "enqueues": ["vcs.sim"],
}


TESTBENCH = r'''`timescale 1ns/1ps
module BBSimVcsHarness;
  reg clock = 1'b0;
  reg reset = 1'b1;
  integer timeout_ns = 100000000;

  BBSimHarness dut (.clock(clock), .reset(reset));

  always #0.5 clock = ~clock;

  initial begin
    if ($value$plusargs("timeout-ns=%d", timeout_ns)) begin end
    repeat (10) @(posedge clock);
    reset = 1'b0;
    #(timeout_ns);
    $fatal(1, "VCS simulation timed out after %0d ns", timeout_ns);
  end

  initial begin
    if ($test$plusargs("vpd")) $vcdpluson(0, dut);
    if ($test$plusargs("vcd")) begin : enable_vcd
      string vcd_file;
      if (!$value$plusargs("vcd=%s", vcd_file)) vcd_file = "activity.vcd";
      $dumpfile(vcd_file);
      $dumpvars(0, dut);
    end
  end
endmodule
'''


# The normal BBSim monitor has a Verilator-specific C++ main.  VCS runs from
# a SystemVerilog testbench, so it needs only the memory model plus these DPI
# callbacks.  Trace callbacks are intentionally no-ops; functional execution,
# UART, and SCU exit remain available to every chip using BBSimHarness.
VCS_DPI = r'''#include <cstdint>
#include <cstdio>
#include <vpi_user.h>

extern "C" void dpi_bdb_set_clk(unsigned long long) {}
extern "C" void bbsim_memory_print_stats();

extern "C" void scu_uart_write(std::uint32_t, std::uint32_t ch) {
  std::fputc(static_cast<int>(ch & 0xff), stdout);
  std::fflush(stdout);
}

extern "C" void scu_sim_exit(std::uint32_t hart, std::uint32_t code) {
  std::fprintf(stderr, "[SCU] hart %u: simulation exit code %u\\n", hart, code);
  std::fflush(stderr);
  bbsim_memory_print_stats();
  vpi_control(vpiFinish, static_cast<PLI_INT32>(code));
}

extern "C" void scu_uart_rx_sample(std::uint32_t, std::uint32_t,
                                     std::uint32_t *valid, std::uint32_t *data) {
  *valid = 0;
  *data = 0;
}

extern "C" void dpi_itrace(std::uint32_t, std::uint32_t, std::uint32_t,
                            std::uint32_t, std::uint32_t, std::uint32_t,
                            std::uint32_t, std::uint32_t, std::uint32_t,
                            std::uint32_t, std::uint32_t, std::uint32_t,
                            std::uint32_t, std::uint32_t, std::uint32_t) {}
extern "C" void dpi_mtrace(std::uint32_t, std::uint32_t, std::uint32_t,
                            std::uint32_t, std::uint32_t, std::uint32_t,
                            std::uint32_t, std::uint32_t, std::uint32_t,
                            std::uint32_t, std::uint32_t, std::uint32_t,
                            std::uint32_t, std::uint32_t, std::uint32_t,
                            std::uint32_t) {}
extern "C" void dpi_pmctrace(std::uint32_t, std::uint32_t, std::uint32_t,
                              std::uint32_t) {}
extern "C" void dpi_mem_pmctrace(std::uint32_t, std::uint32_t,
                                  std::uint32_t, std::uint32_t) {}
extern "C" void dpi_mtrace_issue(std::uint32_t, std::uint32_t,
                                  std::uint32_t, std::uint32_t,
                                  std::uint32_t, std::uint32_t) {}
'''


def _pkg_config(flag: str, package: str) -> str:
    result = subprocess.run(["pkg-config", flag, package], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    try:
        chip = require_chip(input_data)
    except ValueError as exc:
        await check_result(ctx, 1, extra_fields={"task": "build", "error": str(exc)}, trace_id=origin_tid)
        return
    if shutil.which("vcs") is None:
        await check_result(
            ctx,
            1,
            extra_fields={"task": "build", "error": "vcs is not on PATH; source the EDA host environment before running bbdev vcs"},
            trace_id=origin_tid,
        )
        return

    bbdir = get_buckyball_path()
    build_dir = rtl_dir(bbdir, chip, "tapeout", input_data.get("output_dir"))
    vsrcs = sorted(
        path for path in (
            glob.glob(f"{build_dir}/**/*.v", recursive=True)
            + glob.glob(f"{build_dir}/**/*.sv", recursive=True)
        )
        if Path(path).name != "BBSimVcsHarness.sv"
    )
    if not vsrcs:
        await check_result(ctx, 1, extra_fields={"task": "build", "error": f"no RTL found under {build_dir}; run bbdev vcs --verilog first"}, trace_id=origin_tid)
        return

    artifact_dir = Path(build_dir) / "vcs"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    testbench = artifact_dir / "BBSimVcsHarness.sv"
    dpi_shim = artifact_dir / "vcs_dpi.cc"
    testbench.write_text(TESTBENCH, encoding="utf-8")
    dpi_shim.write_text(VCS_DPI, encoding="utf-8")

    arch_dir = Path(bbdir) / "arch"
    csrcs = [
        arch_dir / "src/csrc/src/monitor/ioe/BBSimDRAM.cc",
        arch_dir / "src/csrc/src/monitor/ioe/mm.cc",
        arch_dir / "src/csrc/src/monitor/ioe/mm_dramsim3.cc",
        dpi_shim,
    ]
    missing = [str(path) for path in csrcs if not path.is_file()]
    if missing:
        await check_result(ctx, 1, extra_fields={"task": "build", "error": f"missing VCS DPI source(s): {', '.join(missing)}"}, trace_id=origin_tid)
        return

    result_dir = Path(bbdir) / "result"
    cflags = " ".join(
        flag for flag in (
            "-std=c++17",
            f"-I{result_dir / 'include'}",
            f"-I{build_dir}",
            f"-I{arch_dir / 'src/csrc/include'}",
            "",
        ) if flag
    )
    ldflags = " ".join(
        flag for flag in (
            "-ldramsim3 -lz -lstdc++",
            f"-L{result_dir / 'lib'} -Wl,-rpath,{result_dir / 'lib'}",
        ) if flag
    )
    jobs = int(input_data.get("jobs", 16))
    simv = artifact_dir / "simv"
    compile_log = artifact_dir / "vcs_compile.log"
    sources = [*vsrcs, str(testbench), *(str(path) for path in csrcs)]
    command = " ".join(
        [
            "env -u NIX_LDFLAGS -u NIX_CFLAGS_COMPILE -u NIX_LDFLAGS_FOR_TARGET -u NIX_CFLAGS_COMPILE_FOR_TARGET -u CPATH -u LIBRARY_PATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH -u CFLAGS -u CXXFLAGS -u LDFLAGS",
            "vcs -full64 -sverilog -timescale=1ns/1ps -cpp g++ -cc g++ -ld g++",
            "-top BBSimVcsHarness -debug_access+all -hsopt=off",
            f"-j {jobs}",
            f"-Mdir={shlex.quote(str(artifact_dir / 'csrc'))}",
            f"-o {shlex.quote(str(simv))}",
            f"-l {shlex.quote(str(compile_log))}",
            f"+incdir+{shlex.quote(build_dir)}",
            f"-CFLAGS {shlex.quote(cflags)}",
            f"-LDFLAGS {shlex.quote(ldflags)}",
            *(shlex.quote(source) for source in sources),
        ]
    )
    result = await stream_run_logger_async(
        cmd=command,
        logger=ctx.logger,
        cwd=artifact_dir,
        stdout_prefix="vcs build",
        stderr_prefix="vcs build",
    )
    await check_result(
        ctx,
        result.returncode,
        continue_run=input_data.get("from_run_workflow", False),
        extra_fields={"task": "build", "simv": str(simv), "build_dir": build_dir, "compile_log": str(compile_log)},
        trace_id=origin_tid,
    )
    if result.returncode == 0 and input_data.get("from_run_workflow"):
        await ctx.enqueue({"topic": "vcs.sim", "data": {**input_data, "output_dir": build_dir, "_trace_id": origin_tid}})
