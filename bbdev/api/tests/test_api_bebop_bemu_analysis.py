from datetime import datetime
from pathlib import Path
import shlex
import shutil
import sys
from uuid import uuid4

from _api_test_helper import run_bbdev_case

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.path import get_buckyball_path, log_dir

run_log = Path(
    log_dir(
        get_buckyball_path(),
        "pebble",
        "verilog",
        datetime.now().strftime("%Y-%m-%d-%H-%M"),
        "bemu",
        f"analysis-test-{uuid4().hex}",
    )
)
run_log.mkdir(parents=True)
try:
    shutil.copyfile(
        Path(__file__).parent / "fixtures/bemu-analysis/bdb.ndjson",
        run_log / "bdb.ndjson",
    )
    arguments = f"--design pebble --run {run_log.name} --itrace --mtrace"
    run_bbdev_case(f"bbdev bebop-bemu --analysis {shlex.quote(arguments)}")
finally:
    shutil.rmtree(run_log)
