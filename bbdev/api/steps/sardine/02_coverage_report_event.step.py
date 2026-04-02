import os
import glob
import sys

from motia import FlowContext, queue

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result, get_origin_trace_id

config = {
    "name": "coverage report",
    "description": "merge coverage data and generate annotated source + lcov HTML report",
    "flows": ["sardine"],
    "triggers": [queue("sardine.coverage_report")],
    "enqueues": [],
}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = get_buckyball_path()
    log_dir = f"{bbdir}/arch/log"
    coverage_dir = f"{bbdir}/bb-tests/sardine/reports/coverage"
    os.makedirs(coverage_dir, exist_ok=True)

    # ==================================================================================
    # Find coverage.dat files from this run only (created after run_start_time)
    # ==================================================================================
    all_dat_files = glob.glob(f"{log_dir}/*/coverage.dat")
    run_start_time = input_data.get("run_start_time", 0)
    dat_files = [f for f in all_dat_files if os.path.getmtime(f) >= run_start_time]
    if not dat_files:
        ctx.logger.error("No coverage .dat files found", {"dir": coverage_dir})
        success_result, failure_result = await check_result(
            ctx,
            returncode=1,
            continue_run=False,
            extra_fields={"task": "coverage_report", "error": "No .dat files found"},
            trace_id=origin_tid,
        )
        return

    ctx.logger.info(f"Found {len(dat_files)} coverage data files")

    # ==================================================================================
    # Merge all .dat files
    # ==================================================================================
    merged_dat = f"{coverage_dir}/merged.dat"
    if len(dat_files) == 1:
        # Single file, just copy
        import shutil
        shutil.copy2(dat_files[0], merged_dat)
    else:
        merge_cmd = f"verilator_coverage -write {merged_dat} {' '.join(dat_files)}"
        result = stream_run_logger(
            cmd=merge_cmd,
            logger=ctx.logger,
            cwd=bbdir,
            stdout_prefix="coverage merge",
            stderr_prefix="coverage merge",
        )
        if result.returncode != 0:
            await check_result(
                ctx,
                returncode=result.returncode,
                continue_run=False,
                extra_fields={"task": "coverage_report", "error": "merge failed"},
                trace_id=origin_tid,
            )
            return

    # ==================================================================================
    # Generate annotated source report
    # ==================================================================================
    annotate_dir = f"{coverage_dir}/annotated"
    annotate_cmd = f"verilator_coverage --annotate {annotate_dir} {merged_dat}"
    result = stream_run_logger(
        cmd=annotate_cmd,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="coverage annotate",
        stderr_prefix="coverage annotate",
    )
    if result.returncode != 0:
        await check_result(
            ctx,
            returncode=result.returncode,
            continue_run=False,
            extra_fields={"task": "coverage_report", "error": "annotate failed"},
            trace_id=origin_tid,
        )
        return

    # ==================================================================================
    # Generate lcov info + HTML report
    # ==================================================================================
    lcov_info = f"{coverage_dir}/merged.info"
    html_dir = f"{coverage_dir}/html"

    lcov_cmd = f"verilator_coverage -write-info {lcov_info} {merged_dat}"
    result = stream_run_logger(
        cmd=lcov_cmd,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="coverage lcov",
        stderr_prefix="coverage lcov",
    )
    if result.returncode != 0:
        await check_result(
            ctx,
            returncode=result.returncode,
            continue_run=False,
            extra_fields={"task": "coverage_report", "error": "lcov export failed"},
            trace_id=origin_tid,
        )
        return

    genhtml_cmd = f"genhtml {lcov_info} -o {html_dir}"
    result = stream_run_logger(
        cmd=genhtml_cmd,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="coverage html",
        stderr_prefix="coverage html",
    )
    if result.returncode != 0:
        await check_result(
            ctx,
            returncode=result.returncode,
            continue_run=False,
            extra_fields={"task": "coverage_report", "error": "genhtml failed"},
            trace_id=origin_tid,
        )
        return

    # ==================================================================================
    # Return result
    # ==================================================================================
    ctx.logger.info(f"Coverage report generated: {html_dir}/index.html")
    success_result, failure_result = await check_result(
        ctx,
        returncode=0,
        continue_run=False,
        extra_fields={
            "task": "coverage_report",
            "coverage_dir": coverage_dir,
            "annotate_dir": annotate_dir,
            "html_dir": html_dir,
            "dat_count": len(dat_files),
        },
        trace_id=origin_tid,
    )

    return
