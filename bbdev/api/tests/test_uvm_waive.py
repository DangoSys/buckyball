import sys
from pathlib import Path

API = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API / "steps" / "uvm"))

from scripts.waive import apply_waivers


def test_waivers_are_target_agnostic_and_idempotent(tmp_path):
    rtl = tmp_path / "rtl"
    rtl.mkdir()
    source = rtl / "Dut.sv"
    source.write_text("module Dut;\n  wire _GEN = 1'b0;\nendmodule\n")

    assert apply_waivers(rtl) == 1
    assert source.read_text() == (
        "module Dut;\n"
        "//VCS coverage off\n"
        "  wire _GEN = 1'b0;\n"
        "//VCS coverage on\n"
        "endmodule\n"
    )
    assert apply_waivers(rtl) == 0
