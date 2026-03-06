import os
import glob
import sys

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger
from utils.event_common import check_result

config = {
    "type": "event",
    "name": "coverage report",
    "description": "merge coverage data and generate annotated source + lcov HTML report",
    "subscribes": ["sardine.coverage_report"],
    "emits": [],
    "flows": ["sardine"],
}


async def handler(data, context):
    bbdir = get_buckyball_path()
    log_dir = f"{bbdir}/arch/log"
    coverage_dir = f"{bbdir}/bb-tests/sardine/reports/coverage"
    os.makedirs(coverage_dir, exist_ok=True)

    # ==================================================================================
    # Find coverage.dat files from this run only (created after run_start_time)
    # ==================================================================================
    all_dat_files = glob.glob(f"{log_dir}/*/coverage.dat")
    run_start_time = data.get("run_start_time", 0)
    dat_files = [f for f in all_dat_files if os.path.getmtime(f) >= run_start_time]
    if not dat_files:
        context.logger.error("No coverage .dat files found", {"dir": coverage_dir})
        success_result, failure_result = await check_result(
            context,
            returncode=1,
            continue_run=False,
            extra_fields={"task": "coverage_report", "error": "No .dat files found"},
        )
        return

    context.logger.info(f"Found {len(dat_files)} coverage data files")

    # ==================================================================================
    # Merge all .dat files
    # ==================================================================================
    merged_dat = f"{coverage_dir}/merged.dat"
    if len(dat_files) == 1:
        # Single file, just copy
        import shutil
        shutil.copy2(dat_files[0], merged_dat)
    else:
        merge_cmd = f"verilator_coverage --merge {merged_dat} {' '.join(dat_files)}"
        result = stream_run_logger(
            cmd=merge_cmd,
            logger=context.logger,
            cwd=bbdir,
            stdout_prefix="coverage merge",
            stderr_prefix="coverage merge",
        )
        if result.returncode != 0:
            await check_result(
                context,
                returncode=result.returncode,
                continue_run=False,
                extra_fields={"task": "coverage_report", "error": "merge failed"},
            )
            return

    # ==================================================================================
    # Generate annotated source report
    # ==================================================================================
    annotate_dir = f"{coverage_dir}/annotated"
    annotate_cmd = f"verilator_coverage --annotate {annotate_dir} {merged_dat}"
    result = stream_run_logger(
        cmd=annotate_cmd,
        logger=context.logger,
        cwd=bbdir,
        stdout_prefix="coverage annotate",
        stderr_prefix="coverage annotate",
    )
    if result.returncode != 0:
        await check_result(
            context,
            returncode=result.returncode,
            continue_run=False,
            extra_fields={"task": "coverage_report", "error": "annotate failed"},
        )
        return

    # ==================================================================================
    # Generate lcov info + HTML report
    # ==================================================================================
    lcov_info = f"{coverage_dir}/merged.info"
    html_dir = f"{coverage_dir}/html"

    lcov_cmd = f"verilator_coverage --write-info {lcov_info} {merged_dat}"
    result = stream_run_logger(
        cmd=lcov_cmd,
        logger=context.logger,
        cwd=bbdir,
        stdout_prefix="coverage lcov",
        stderr_prefix="coverage lcov",
    )
    if result.returncode != 0:
        await check_result(
            context,
            returncode=result.returncode,
            continue_run=False,
            extra_fields={"task": "coverage_report", "error": "lcov export failed"},
        )
        return

    genhtml_cmd = f"genhtml {lcov_info} -o {html_dir}"
    result = stream_run_logger(
        cmd=genhtml_cmd,
        logger=context.logger,
        cwd=bbdir,
        stdout_prefix="coverage html",
        stderr_prefix="coverage html",
    )
    if result.returncode != 0:
        await check_result(
            context,
            returncode=result.returncode,
            continue_run=False,
            extra_fields={"task": "coverage_report", "error": "genhtml failed"},
        )
        return

    # ==================================================================================
    # Return result
    # ==================================================================================
    context.logger.info(f"Coverage report generated: {html_dir}/index.html")
    success_result, failure_result = await check_result(
        context,
        returncode=0,
        continue_run=False,
        extra_fields={
            "task": "coverage_report",
            "coverage_dir": coverage_dir,
            "annotate_dir": annotate_dir,
            "html_dir": html_dir,
            "dat_count": len(dat_files),
        },
    )

    return
