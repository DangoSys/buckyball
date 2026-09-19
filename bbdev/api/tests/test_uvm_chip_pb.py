import sys
from pathlib import Path

API = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API / "steps" / "config" / "scripts"))
sys.path.insert(0, str(API / "steps" / "uvm"))

import chip_pb2
from scripts.uvm_common import ball_domain, load_ip, selected_mappings, vcs_defines


def _chip():
    c = chip_pb2.Chip()
    core = c.cores.add()
    m = core.balldomain.mappings.add()
    m.ball_id = 5
    m.ball_dir = "lut"
    m.in_bw = 2
    m.out_bw = 1
    isa = core.balldomain.isa.add()
    isa.mnemonic = "LUT"
    isa.funct7 = 66
    isa.bid = 5
    return c


def test_defines():
    c = _chip()
    d = ball_domain(c)
    m = selected_mappings(d, "lut")[0]
    defs = vcs_defines(d, m, 4096)
    assert defs == [
        "+define+BB_IN_BW=2",
        "+define+BB_OUT_BW=1",
        "+define+BB_MMIO_READ_BW=0",
        "+define+BB_MMIO_WRITE_BW=0",
        "+define+BB_BANK_ADDR_W=12",
        "+define+LUT_FUNCT7=66",
    ]


def test_registered_ip():
    root = str(Path(__file__).resolve().parents[3])
    axis = load_ip(root, "axis")
    assert axis["mill_module"] == "axis"
    assert [target["name"] for target in axis["targets"]] == ["arbiter", "master", "slave"]


def test_unknown_ball():
    try:
        selected_mappings(ball_domain(_chip()), "transpose")
    except ValueError as e:
        assert "transpose" in str(e)
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    test_defines()
    test_registered_ip()
    test_unknown_ball()
    print("ok")
