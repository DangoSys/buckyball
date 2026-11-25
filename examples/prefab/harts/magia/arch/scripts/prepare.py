"""Prepare the pinned MAGIA mesh sources for a Buckyball RTL build."""

import argparse
from pathlib import Path
import re
import subprocess
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
args = parser.parse_args()
hart = Path(__file__).resolve().parents[2]
repo = hart / "thirdparty/MAGIA"
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)


def bender(*arguments):
    return subprocess.check_output(["bender", *arguments], cwd=repo, text=True).strip()


idma = bender("path", "idma")
subprocess.run(
    [
        "make",
        "-C",
        idma,
        "idma_hw_all",
        "IDMA_ADD_IDS=rw_axi_rw_obi",
        f"BENDER=bender -d {repo}",
    ],
    check=True,
)
targets = [
    "rtl",
    "snitch_cluster",
    "spatz",
    "redmule_hwpe",
    "tech_cells_generic_include_deprecated",
    "verilator",
    "rtl_sim",
    "magia_dv",
    "cv32e40p_include_tracer",
]
defines = [
    "CV32E40P",
    "ZFINX_CTRL=1",
    "PULP_ZFINX_DEF",
    "ZFINX_CLUSTER=1",
    "TARGET_SPATZ",
    "SPATZ_RVD=0",
    "SPATZ_VLEN=256",
    "SPATZ_NRVREG=32",
    "SPATZ_NR_VRF_BANKS=4",
    "SPATZ_N_IPU=1",
    "SPATZ_N_FPU=4",
    "SPATZ_NR_PARALLEL_INSTR=4",
    "SPATZ_XDIVSQRT=0",
    "SPATZ_XDMA=0",
    "SPATZ_RVF=1",
    "SPATZ_RVV=1",
    "SYNTHESIS",
]
arguments = ["script", "verilator", "-e", "ibex", "-e", "common_verification"]
for target in targets:
    arguments += ["-t", target]
for define in defines:
    arguments += ["-D", define]
files = bender(*arguments)
filtered = output / "magia.f"
subprocess.run(
    [
        "python3",
        str(repo / "verilator/scripts/filter_filelist.py"),
        "--output",
        str(filtered),
    ],
    input=files,
    text=True,
    cwd=repo,
    check=True,
)
lines = [
    f"+incdir+{repo / 'target/verilator/src'}",
    f"+incdir+{repo / 'target/verilator/include'}",
    f"+incdir+{Path(bender('path', 'fractal_sync')) / 'hw'}",
]
for line in filtered.read_text().splitlines():
    # Keep the CPU wrapper while excluding its file-writing tracer.
    if line.startswith(("+define+CV32E40P_TRACE_EXECUTION", "+define+HART_TRACES")):
        continue
    path = Path(line)
    if path.name == "tc_sram.sv":
        lines.append(str(hart / "arch/tile/Sram.sv"))
        continue
    if path.name in {"tb_memory_regbus.sv", "tb_memory_axi.sv"}:
        continue
    if path == repo / "hw/mesh/noc/magia_noc_pkg.sv":
        lines.append(str(hart / "arch/NoCPackage.sv"))
        continue
    if path == repo / "hw/tile/idma_ctrl_mm.sv":
        lines.append(str(hart / "arch/tile/IdmaControl.sv"))
        continue
    if path.name == "redmule_streamer.sv":
        lines.append(str(hart / "arch/tile/RedmuleStreamer.sv"))
        continue
    if path == repo / "target/verilator/src/magia_tile_hier.sv":
        lines.append(str(hart / "arch/Tile.sv"))
        continue
    if path == repo / "hw/mesh/magia.sv":
        lines.append(str(hart / "arch/Mesh.sv"))
        continue
    if path == repo / "hw/mesh/noc/floo_axi_nw_mesh_4x4_noc.sv":
        lines.append(str(hart / "arch/NoC.sv"))
        continue
    if path.is_relative_to(repo / "target/sim"):
        continue
    if (
        path.is_relative_to(repo / "target/verilator")
        and path.name != "magia_tile_hier.sv"
    ):
        continue
    if not line.startswith(("+", "-")):
        original = path.read_text()
        text = original
        if path.name == "hci_core_split.sv":
            # These handshake temporaries are scalar bits, not aggregates.
            text = text.replace(
                "'{default: {tcdm_initiator_req[ii]}}", "tcdm_initiator_req[ii]"
            )
            text = text.replace(
                "'{default: {tcdm_initiator_r_ready[ii]}}", "tcdm_initiator_r_ready[ii]"
            )
        if path == repo / "hw/tile/magia_tile_pkg.sv":
            # DC does not take index keys in assignment patterns: rule 0 of the
            # four (packed, MSB first) cached regions, positionally.
            text = text.replace(
                "CachedRegion: '{\n      0: '{base: INSTRRAM_ADDR_START, mask: INSTRRAM_PMA_MASK},\n"
                "      default: '0\n    },",
                "CachedRegion: '{'0, '0, '0, '{base: INSTRRAM_ADDR_START, mask: INSTRRAM_PMA_MASK}},",
            )
        if path == repo / "hw/tile/magia_tile.sv":
            # HART_TRACES is disabled, so the controller has no tracer wrapper.
            text = text.replace(".cv32e40p_top_i.hart_i", ".hart_i")
        if path.name == "event_unit_top.sv":
            # VSYN checks interface-array ports against the declaration defaults.
            # Match the single-core MAGIA event unit and its two barriers.
            text = text.replace("parameter NB_CORES = 8", "parameter NB_CORES = 1")
            text = text.replace("parameter NB_BARR  = 8", "parameter NB_BARR  = 2")
            text = text.replace(
                "periph_int_bus[NB_CORES+NB_BARR+2:0]()", "periph_int_bus[5:0]()"
            )
            text = text.replace(
                "demux_int_bus_core[NB_CORES-1:0]()", "demux_int_bus_core[0:0]()"
            )
            text = text.replace(
                "demux_int_bus_barrier[NB_BARR-1:0]()", "demux_int_bus_barrier[1:0]()"
            )
        if path == repo / "hw/tile/magia_event_unit.sv":
            text = text.replace(
                "eu_direct_link[NB_CORES-1:0]()", "eu_direct_link[0:0]()"
            )
        if path.name == "hci_router.sv":
            # The mesh connects each router to the tile's 32 memory banks.
            text = text.replace("NB_OUT_CHAN = 8", "NB_OUT_CHAN = 32")
        if path.name == "hci_arbiter_tree.sv":
            text = text.replace("NB_CHAN = 16", "NB_CHAN = 32")
            # VSYN sizes the input interface array before applying overrides.
            # Keep the HWPE tree at one requester and specialize DMA and core.
            tree = text
            for unit, requests in (("dma", 4), ("core", 6)):
                text += "\n" + tree.replace(
                    "module hci_arbiter_tree", f"module magia_{unit}_arbiter_tree", 1
                ).replace("NB_REQUESTS = 1", f"NB_REQUESTS = {requests}", 1)
        if path == repo / "hw/tile/local_interconnect.sv":
            for unit in ("dma", "core"):
                text = re.sub(
                    rf"hci_arbiter_tree(?= #\(\s*\.NB_REQUESTS\s*\(\s*N_{unit.upper()}\b)",
                    f"magia_{unit}_arbiter_tree",
                    text,
                )
        if path.name == "cv32e40p_sim_clock_gate.sv":
            text = text.replace(
                "always_latch begin\n    if (clk_i == 1'b0) clk_en <= en_i | scan_cg_en_i;\n  end",
                "always_ff @(negedge clk_i) clk_en <= en_i | scan_cg_en_i;",
            )
        if path == repo / "hw/tile/obi_slave_fsync.sv":
            text = text.replace(
                "tc_clk_gating i_sync_clock_gating",
                "MagiaClockGate i_sync_clock_gating",
            )
        if path.name == "redmule_top.sv":
            text = text.replace(
                "tc_clk_gating i_acc_clock_gating", "MagiaClockGate i_acc_clock_gating"
            )
        # Preprocessing removes translation pragmas, so apply them as guards first.
        synthesis = re.sub(
            r"(?m)^\s*//\s*(?:pragma|synopsys|synthesis)\s+translate_(off|on)\b[^\n]*$",
            lambda match: "`ifndef SYNTHESIS" if match[1] == "off" else "`endif",
            text,
        )
        if synthesis != original:
            source = output / "synthesis" / path.relative_to(repo)
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(synthesis)
            line = str(source)
    lines.append(line)
lines += [
    str(hart / "arch/ClockGate.sv"),
    str(hart / "arch/MagiaMemory.sv"),
    str(hart / "arch/MeshAXI.sv"),
]
filtered.write_text("\n".join(lines) + "\n")
source = output / "MagiaSources.sv"
entry = (hart / "arch/MeshAXI.sv").read_text()
start = entry.index("module MagiaMeshAXI")
(output / "MagiaMeshAXI.stub").write_text(
    entry[start : entry.index(");", start) + 2] + "\nendmodule\n"
)
with source.open("w") as stream:
    subprocess.run(
        ["verilator", "-E", "-P", "-f", str(filtered)],
        cwd=repo,
        stdout=stream,
        check=True,
    )
# P2E takes the mesh as it is synthesized: the same sources without the simulation-only parts.
with (output / "MagiaMeshAXI.native").open("w") as stream:
    subprocess.run(
        ["verilator", "-E", "-P", "-f", str(filtered), "-UVERILATOR"],
        cwd=repo,
        stdout=stream,
        check=True,
    )
shutil.copyfile(hart / "arch/hierarchy.vlt", output / "hierarchy.vlt")

shutil.copyfile(hart / "arch/verilator.f", output / "verilator.f")

shutil.copyfile(hart / "arch/hierarchy_params.v", output / "hierarchy_params.v")
