import os
import shlex
import subprocess
import sys
import tomllib
from datetime import datetime
from pathlib import Path

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
bebop_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if bebop_path not in sys.path:
    sys.path.insert(0, bebop_path)

from utils.workload_manifest import resolve_workload_toml
from utils.event_common import check_result, get_origin_trace_id, require_chip
from utils.path import get_buckyball_path, log_dir, rtl_dir, workloads_output_root
from utils.search_workload import search_workload
from utils.stream_run import stream_run_logger_async

config = {"name": "bebop-vcs-batch", "flows": ["bebop"], "triggers": [queue("bebop.vcs.batch")], "enqueues": []}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    try:
        chip = require_chip(input_data)
        bbdir = get_buckyball_path()
        test_type = input_data.get("test", "elf-tests")
        manifest = resolve_workload_toml(chip, "verilator", test_type, bbdir)
    except ValueError as error:
        await check_result(ctx, 1, extra_fields={"error": str(error)}, trace_id=origin_tid)
        return
    simv = Path(rtl_dir(bbdir, chip, "verilog")) / "vcs" / "simv"
    if not simv.is_file():
        await check_result(ctx, 1, extra_fields={"error": "missing VCS simv: run bebop-vcs --verilog and --build first"}, trace_id=origin_tid)
        return
    tests = tomllib.loads(Path(manifest).read_text())["workloads"]["tests"]
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    report = Path(log_dir(bbdir, chip, "verilog", stamp, "bebop-vcs", test_type))
    report.mkdir(parents=True)
    libc = subprocess.check_output(["g++", "-print-file-name=libc.so.6"], text=True).strip()
    ball_names = [path.name for path in (Path(bbdir) / "examples" / "balls").iterdir() if path.is_dir()]
    counts = {ball: [0, 0] for ball in ball_names}
    passed = 0
    for name in tests:
        ball = next((item for item in ball_names if f"-ctest-{item}" in name), "framework")
        counts.setdefault(ball, [0, 0])[1] += 1
        binary = search_workload(workloads_output_root(bbdir), name)
        if binary is None:
            continue
        command = " ".join(("env", f"LD_LIBRARY_PATH={shlex.quote(str(Path(libc).parent))}", shlex.quote(str(simv)), f"+elf={shlex.quote(binary)}", "+batch", "+trace=none", "+no-wave", f"+dramsim_ini_dir={shlex.quote(str(Path(bbdir) / 'result/share/dramsim3/configs'))}"))
        result = await stream_run_logger_async(cmd=command, logger=ctx.logger, cwd=str(simv.parent), stdout_prefix="bebop vcs batch", stderr_prefix="bebop vcs batch")
        if result.returncode == 0:
            counts[ball][0] += 1
            passed += 1
    index = report / "test-index.txt"
    index.write_text("ball_dir vcs\n" + "\n".join(f"{ball} {ok}/{total}" for ball, (ok, total) in sorted(counts.items()) if total) + "\n")
    await check_result(ctx, 0 if passed == len(tests) else 1, extra_fields={"task": "batch", "passed": passed, "total": len(tests), "test_index": str(index)}, trace_id=origin_tid)
