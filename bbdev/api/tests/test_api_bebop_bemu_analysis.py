from pathlib import Path
import shutil
import tempfile

from _api_test_helper import run_bbdev_case

src = Path(__file__).resolve().parent / "fixtures" / "bemu-analysis"
dst = Path(tempfile.mkdtemp()) / "bemu-analysis"
shutil.copytree(src, dst)
run_bbdev_case(
    f"bbdev bebop-bemu --analysis '--chip pebble --log-dir {dst} --itrace --mtrace'"
)
