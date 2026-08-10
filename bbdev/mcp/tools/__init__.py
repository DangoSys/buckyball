"""Register all MCP tools."""

from __future__ import annotations

from . import validate
from . import task_status
from . import task_cancel
from . import compiler_build
from . import workload_clean
from . import workload_build
from . import bemu_sim
from . import bemu_batch
from . import bebop_verilator_clean
from . import bebop_verilator_verilog
from . import bebop_verilator_build
from . import bebop_verilator_sim
from . import bebop_verilator_run
from . import bebop_verilator_batch
from . import difftest_run
from . import verilator_clean
from . import verilator_verilog
from . import verilator_build
from . import verilator_sim
from . import verilator_run
from . import uvm_build
from . import uvm_run
from . import workload_tohex
from . import bebop_p2e_clean
from . import bebop_p2e_verilog
from . import bebop_p2e_buildbitstream
from . import bebop_p2e_runworkload
from . import bebop_p2e_batch
from . import dc_verilog
from . import ip_replace
from . import firesim_enumeratefpgas
from . import firesim_buildbitstream
from . import firesim_infrasetup
from . import firesim_runworkload
from . import kernel_build
from . import yosys_run
from . import yosys_verilog
from . import yosys_synth

MODULES = [
    validate,
    task_status,
    task_cancel,
    compiler_build,
    workload_clean,
    workload_build,
    bemu_sim,
    bemu_batch,
    bebop_verilator_clean,
    bebop_verilator_verilog,
    bebop_verilator_build,
    bebop_verilator_sim,
    bebop_verilator_run,
    bebop_verilator_batch,
    difftest_run,
    verilator_clean,
    verilator_verilog,
    verilator_build,
    verilator_sim,
    verilator_run,
    uvm_build,
    uvm_run,
    workload_tohex,
    bebop_p2e_clean,
    bebop_p2e_verilog,
    bebop_p2e_buildbitstream,
    bebop_p2e_runworkload,
    bebop_p2e_batch,
    dc_verilog,
    ip_replace,
    firesim_enumeratefpgas,
    firesim_buildbitstream,
    firesim_infrasetup,
    firesim_runworkload,
    kernel_build,
    yosys_run,
    yosys_verilog,
    yosys_synth,
]


def register_all(mcp) -> None:
    for mod in MODULES:
        mod.register(mcp)
