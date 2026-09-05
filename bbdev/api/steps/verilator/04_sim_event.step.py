import os
import shlex
import struct
import subprocess
import sys
from datetime import datetime

from motia import FlowContext, queue

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.event_common import require_chip
from utils.path import get_buckyball_path, log_dir, rtl_dir, workload_tests_root
from utils.stream_run import stream_run_logger_async
from utils.search_workload import search_workload
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "verilator-sim",
    "description": "run simulation",
    "flows": ["verilator"],
    "triggers": [queue("verilator.sim")],
    "enqueues": [],
}


DRAM_BASE = 0x80000000
DRAM_SIZE = 0x10000000


def validate_bbsim_elf(path: str) -> tuple[bool, str]:
    """Validate the ELF layout accepted by BBSimDRAM before starting Verilator."""
    try:
        with open(path, "rb") as binary:
            header = binary.read(64)
            if len(header) != 64:
                return False, "not a complete ELF64 header"
            if header[:4] != b"\x7fELF" or header[4] != 2 or header[5] != 1:
                return False, "requires a little-endian ELF64 binary"

            (_, _, machine, _, _, phoff, _, _, ehsize, phentsize, phnum, _, _, _) = (
                struct.unpack("<16sHHIQQQIHHHHHH", header)
            )
            if machine != 243:
                return False, "requires a RISC-V ELF"
            if ehsize != 64 or phentsize != 56 or phnum == 0:
                return False, "has an unsupported ELF program-header layout"

            file_size = os.fstat(binary.fileno()).st_size
            if phoff + phnum * phentsize > file_size:
                return False, "program headers extend past the end of the file"

            load_segments = 0
            for index in range(phnum):
                binary.seek(phoff + index * phentsize)
                program_header = binary.read(phentsize)
                if len(program_header) != phentsize:
                    return False, "cannot read ELF program headers"
                p_type, _, p_offset, _, p_paddr, p_filesz, p_memsz, _ = struct.unpack(
                    "<IIQQQQQQ", program_header
                )
                if p_type != 1 or p_filesz == 0:
                    continue
                load_segments += 1
                if p_filesz > p_memsz or p_offset + p_filesz > file_size:
                    return False, "has an invalid loadable segment"
                if p_paddr < DRAM_BASE or p_paddr + p_memsz > DRAM_BASE + DRAM_SIZE:
                    return (
                        False,
                        f"load segment {index} at 0x{p_paddr:x} is outside "
                        f"BBSimDRAM [0x{DRAM_BASE:x}, 0x{DRAM_BASE + DRAM_SIZE:x})",
                    )
    except (OSError, struct.error):
        return False, "cannot read a valid ELF64 program-header table"

    if not load_segments:
        return False, "has no loadable ELF segments"
    return True, ""


async def handler(input_data: dict, ctx: FlowContext) -> None:
    # ==================================================================================
    # Get simulation parameters
    # ==================================================================================
    origin_tid = get_origin_trace_id(input_data, ctx)
    try:
        chip = require_chip(input_data)
    except ValueError as error:
        ctx.logger.error(str(error))
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "missing_chip"},
            trace_id=origin_tid,
        )
        return
    bbdir = get_buckyball_path()
    build_dir = rtl_dir(
        bbdir, chip, "verilog", input_data.get("output_dir"),
    )
    ctx.logger.info(f"Using build directory: {build_dir}")

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

    binary_name = input_data.get("binary", "")
    if not binary_name:
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "binary_required"},
            trace_id=origin_tid,
        )
        return
    coverage = input_data.get("coverage", False)

    workload_root = workload_tests_root(bbdir, chip)
    binary_path = search_workload(workload_root, binary_name)
    ctx.logger.info(f"binary_path: {binary_path}")
    if binary_path is None:
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={
                "error": "binary_not_found",
                "binary": binary_name,
                "search_dir": workload_root,
            },
            trace_id=origin_tid,
        )
        return
    valid_elf, elf_error = validate_bbsim_elf(binary_path)
    if not valid_elf:
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={
                "error": "unsupported_elf",
                "binary": binary_path,
                "detail": (
                    "direct Verilator requires a baremetal RISC-V ELF in BBSimDRAM; "
                    f"{elf_error}. Use bebop-verilator for Linux ELFs"
                ),
            },
            trace_id=origin_tid,
        )
        return

    topname = "BBSimHarness"
    run_log = log_dir(bbdir, chip, "verilog", timestamp, "verilator", binary_name, input_data.get("output_dir"))
    waveform_dir = os.path.join(run_log, "waveform")

    os.makedirs(run_log, exist_ok=True)
    no_wave = bool(input_data.get("no-wave", input_data.get("no_wave", False)))
    if not no_wave:
        os.makedirs(waveform_dir, exist_ok=True)

    coverage_flag = ""
    if coverage:
        coverage_dat_path = f"{run_log}/coverage.dat"
        coverage_flag = f"+verilator+coverage+file+{coverage_dat_path}"

    bin_path = f"{build_dir}/obj_dir/V{topname}"
    if not os.path.isfile(bin_path):
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"error": "simulator_not_found", "simulator": bin_path},
            trace_id=origin_tid,
        )
        return
    batch = input_data.get("batch", False)

    trace_names = [
        name
        for name in ("itrace", "mtrace", "pmctrace", "ctrace", "banktrace")
        if input_data.get(name, False)
    ]
    trace_arg = ",".join(trace_names) if trace_names else "none"

    log_path    = f"{run_log}/bdb.ndjson"
    stdout_path = f"{run_log}/stdout.log"
    meta_path   = f"{run_log}/sim_meta.txt"
    fst_path    = f"{waveform_dir}/waveform.fst"

    # ==================================================================================
    # Execute simulation
    # BBSimHarness uses +elf= for ELF loading (via BBSimDRAM.cc / libelf)
    # No fesvr, no +loadmem_addr needed
    #
    # disasm.log: only stderr -> spike-dasm (Rocket commit printf is stderr here;
    # merging stdout with 2>&1 can break: full stdio buffering + non-DASM bytes).
    # BDB_SIM_META moves NDJSON banner to sim_meta.txt so it does not pollute disasm.
    # ==================================================================================
    result_lib = f"{bbdir}/result/lib"
    def _gcc_lib_dir(soname: str) -> str:
        printed = subprocess.run(
            ["g++", f"-print-file-name={soname}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return os.path.dirname(os.path.realpath(printed.stdout.strip()))
    lib_dirs = [result_lib, _gcc_lib_dir("liblz4.so"), _gcc_lib_dir("libstdc++.so")]
    ld_lib_path = ":".join(dict.fromkeys(lib_dirs))
    ctx.logger.info(f"LD_LIBRARY_PATH prefix: {ld_lib_path}")
    sim_args = [
        bin_path,
        "+permissive",
        f"+elf={binary_path}",
    ]
    if batch:
        sim_args.append("+batch")
    if coverage_flag:
        sim_args.append(coverage_flag)
    if no_wave:
        sim_args.append("+no-wave")
    else:
        sim_args.append(f"+fst={fst_path}")
    sim_args.extend([
        f"+log={log_path}",
        f"+stdout={stdout_path}",
        f"+trace={trace_arg}",
        "+permissive-off",
    ])
    sim_cmd = (
        f"export LD_LIBRARY_PATH={shlex.quote(ld_lib_path)}:$LD_LIBRARY_PATH; "
        f"export BDB_SIM_META={shlex.quote(meta_path)}; "
        f"{shlex.join(sim_args)} "
        f"2> >(spike-dasm > {shlex.quote(os.path.join(run_log, 'disasm.log'))})"
    )
    script_dir = os.path.dirname(__file__)

    result = await stream_run_logger_async(
        cmd=sim_cmd,
        logger=ctx.logger,
        cwd=script_dir,
        stdout_prefix="verilator sim",
        stderr_prefix="verilator sim",
        executable="bash",
    )
    success_result, failure_result = await check_result(
        ctx, returncode=result.returncode, continue_run=True, trace_id=origin_tid,
    )
    if failure_result:
        ctx.logger.error("sim failed", failure_result)
        return

    # ==================================================================================
    # Return simulation result
    # ==================================================================================
    extra_fields = {
        "task": "sim",
        "binary": binary_path,
        "log_dir": run_log,
        "waveform_dir": None if no_wave else waveform_dir,
        "timestamp": timestamp,
        "sim_meta": meta_path,
    }
    if coverage:
        extra_fields["coverage_dat"] = coverage_dat_path

    await check_result(
        ctx,
        result.returncode,
        continue_run=False,
        extra_fields=extra_fields,
        trace_id=origin_tid,
    )

    return
