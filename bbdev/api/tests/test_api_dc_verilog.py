import json
from pathlib import Path

from _api_test_helper import run_bbdev_case

run_bbdev_case("bbdev dc --verilog '--chip toy'")

root = Path(__file__).resolve().parents[3]
output_dir = root / "arch" / "build" / "toy" / "sims.verilator.BuckyballToyVerilatorConfig"
manifest = json.loads((output_dir / "ip-replace" / "sram_manifest.json").read_text())
source_list = (output_dir / "ip-replace" / "dc_sources.list").read_text().splitlines()

assert manifest["top_module"] == "DigitalTop"
assert manifest["memories"]
assert any(path.endswith("/DigitalTop.sv") for path in source_list)
assert not any(path.endswith("/BBSimHarness.sv") for path in source_list)
