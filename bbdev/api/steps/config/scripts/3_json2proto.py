#!/usr/bin/env python3
"""Build Chip protobuf from config.json + derived.json."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import chip_pb2 as pb  # noqa: E402


def _load_derive():
    path = _SCRIPTS / "2_parameter_derivation.py"
    spec = importlib.util.spec_from_file_location("parameter_derivation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_derive = _load_derive()


def _rel(bbdir: Path, path: str | Path) -> str:
    p = Path(path)
    if p.is_absolute():
        return p.resolve().relative_to(bbdir.resolve()).as_posix()
    return p.as_posix()


def _ball_dir(ball_class: str) -> str:
    parts = ball_class.split(".")
    if len(parts) < 3 or parts[0] != "examples" or parts[1] != "balls":
        raise ValueError(f"bad ballClass: {ball_class}")
    return parts[2]


def _raw_core_list(designs: dict[str, Any]) -> list[dict[str, Any]]:
    return list(_derive.iter_cores(designs))


def _raw_tile_proto(designs: dict[str, Any]) -> dict[str, Any]:
    return _derive.iter_topology_tiles(designs)[0]


def _fill_bank(msg: pb.BankConfig, d: dict[str, Any]) -> None:
    msg.num = d["num"]
    msg.width = d["width"]
    msg.entries = d["entries"]
    msg.mask_len = d["maskLen"]
    msg.channel = d["channel"]


def _fill_mem(msg: pb.MemDomainConfig, d: dict[str, Any], bbdir: Path) -> None:
    msg.source_path = _rel(bbdir, d["_file"])
    _fill_bank(msg.bank, d["bank"])
    dma = d["dma"]
    msg.dma.n_xacts = dma["nXacts"]
    msg.dma.burst_max_bytes = dma["burstMaxBytes"]
    msg.dma.bus_width = dma["busWidth"]
    msg.dma.max_in_flight_mem_reqs = dma["maxInFlightMemReqs"]
    msg.tlb.size = d["tlb"]["size"]
    msg.tma.read_channel = d["tma"]["readChannel"]
    msg.tma.write_channel = d["tma"]["writeChannel"]
    mmio = d["mmio"]
    msg.mmio.enable = mmio["enable"]
    msg.mmio.bank_num = mmio["bankNum"]
    msg.mmio.bank_entries = mmio["bankEntries"]
    msg.mmio.bank_width = mmio["bankWidth"]
    msg.mmio.read_width = mmio["readWidth"]
    msg.mem.addr_len = d["mem"]["addrLen"]


def _fill_ball(msg: pb.BallDomain, d: dict[str, Any], bbdir: Path) -> None:
    msg.source_path = _rel(bbdir, d["_file"])
    msg.ball_num = d["ballNum"]
    for m in d["ballIdMappings"]:
        e = msg.mappings.add()
        e.ball_id = m["ballId"]
        e.ball_name = m["ballName"]
        e.ball_class = m["ballClass"]
        e.ball_dir = _ball_dir(m["ballClass"])
        e.config_path = _rel(bbdir, m["config"]["_file"])
        e.in_bw = m["inBW"]
        e.out_bw = m["outBW"]
        if "mmioReadBW" in m:
            e.mmio_read_bw = m["mmioReadBW"]
        if "mmioWriteBW" in m:
            e.mmio_write_bw = m["mmioWriteBW"]
    for item in d["ballISA"]:
        e = msg.isa.add()
        e.mnemonic = item["mnemonic"]
        e.funct7 = item["funct7"]
        e.bid = item["bid"]


def _fill_rocket(msg: pb.RocketCoreConfig, d: dict[str, Any]) -> None:
    msg.x_len = d["xLen"]
    msg.pg_levels = d["pgLevels"]
    msg.use_vm = d["useVM"]
    msg.use_zba = d["useZba"]
    msg.use_zbb = d["useZbb"]
    msg.use_zbs = d["useZbs"]
    msg.have_c_flush = d["haveCFlush"]
    md = d["mulDiv"]
    msg.mul_div.enable = md["enable"]
    msg.mul_div.mul_unroll = md["mulUnroll"]
    msg.mul_div.mul_early_out = md["mulEarlyOut"]
    msg.mul_div.div_early_out = md["divEarlyOut"]
    fpu = d["fpu"]
    msg.fpu.enable = fpu["enable"]
    msg.fpu.min_f_len = fpu["minFLen"]
    msg.fpu.f_len = fpu["fLen"]
    dc = d["dcache"]
    msg.dcache.n_sets = dc["nSets"]
    msg.dcache.n_ways = dc["nWays"]
    msg.dcache.n_mshrs = dc["nMSHRs"]
    ic = d["icache"]
    msg.icache.n_sets = ic["nSets"]
    msg.icache.n_ways = ic["nWays"]
    btb = d["btb"]
    msg.btb.enable = btb["enable"]
    msg.btb.n_entries = btb["nEntries"]
    msg.btb.n_ras = btb["nRAS"]


def _fill_frontend(msg: pb.FrontendConfig, d: dict[str, Any], bbdir: Path) -> None:
    msg.source_path = _rel(bbdir, d["_file"])
    msg.rob_entries = d["robEntries"]
    msg.rs_out_of_order_response = d["rsOutOfOrderResponse"]
    msg.bank_id_len = d["bankIdLen"]
    msg.vbank_id_upper_bound = d["vbankIdUpperBound"]
    msg.shared_bank_id_base = d["sharedBankIdBase"]
    msg.iter_len = d["iterLen"]
    msg.sub_rob_enable = d["subRobEnable"]
    msg.sub_rob_depth = d["subRobDepth"]


def _fill_gp(msg: pb.GpDomainConfig, d: dict[str, Any], bbdir: Path) -> None:
    msg.source_path = _rel(bbdir, d["_file"])
    msg.lane_number = d["laneNumber"]
    msg.chaining_size = d["chainingSize"]
    msg.v_len = d["vLen"]
    msg.d_len = d["dLen"]
    msg.e_len = d["eLen"]
    msg.lane_scale = d["laneScale"]


def _fill_core_param(msg: pb.CoreParamConfig, d: dict[str, Any], bbdir: Path) -> None:
    msg.source_path = _rel(bbdir, d["_file"])
    msg.core_data_bytes = d["coreDataBytes"]
    msg.x_len = d["xLen"]
    msg.vaddr_bits = d["vaddrBits"]
    msg.paddr_bits = d["paddrBits"]
    msg.pg_idx_bits = d["pgIdxBits"]
    msg.n_pmps = d["nPMPs"]


def _fill_core(ci: pb.CoreInstance, raw: dict[str, Any], meta: dict[str, Any], bbdir: Path) -> None:
    ci.index = meta["index"]
    ci.role = meta["role"]
    ci.pkg = meta["pkg"]
    ci.config_path = meta["config_path"]
    ci.balldomain_base_dir = meta["balldomain_base_dir"]
    if "balldomain" in raw:
        _fill_ball(ci.balldomain, raw["balldomain"], bbdir)
    if "memdomain" in raw:
        _fill_mem(ci.mem, raw["memdomain"], bbdir)
    if "rocketCore" in raw:
        _fill_rocket(ci.rocket_core, raw["rocketCore"])
    if "frontend" in raw:
        _fill_frontend(ci.frontend, raw["frontend"], bbdir)
    if "gpdomain" in raw:
        _fill_gp(ci.gp_domain, raw["gpdomain"], bbdir)
    if "core" in raw:
        _fill_core_param(ci.core, raw["core"], bbdir)


def _fill_tile(tp: pb.TilePlacement, meta: dict[str, Any], proto: dict[str, Any]) -> None:
    tp.path = meta["path"]
    tp.virtual_bank_count = meta["virtual_bank_count"]
    tp.core_indices.extend(meta["core_indices"])
    tp.mem_ball_channel_num = meta["mem_ball_channel_num"]
    dc = proto["privateDCache"]
    tp.private_dcache.enable = dc["enable"]
    tp.private_dcache.ways = dc["ways"]
    tp.private_dcache.capacity_kb = dc["capacityKB"]
    tp.private_dcache.write_bytes = dc["writeBytes"]
    tp.private_dcache.port_factor = dc["portFactor"]
    tp.private_dcache.mem_cycles = dc["memCycles"]
    sm = proto["sharedMem"]
    tp.shared_mem.enable = sm["enable"]
    tp.shared_mem.entries = sm["entries"]
    tp.shared_mem.input_channels = sm["inputChannels"]
    tp.shared_mem.default_group_count = sm["defaultGroupCount"]
    if "virtualBankCount" in sm:
        tp.shared_mem.virtual_bank_count = sm["virtualBankCount"]
    if "memCycles" in sm:
        tp.shared_mem.mem_cycles = sm["memCycles"]



def fill_chip(config: dict[str, Any], derived: dict[str, Any], bbdir: Path) -> pb.Chip:
    """
    Hardware numbers live in config.json; identities/paths live in derived.json.
    chip.pb is a curated schema, so we pick from both instead of dumping either file.

    Core/tile lists vs templates are already expanded in step2 — import that walker,
    do not write a second one.

    Attention: All tiles on a chip must match, so SRAM/dcache
    is copied from the first tile onto every TilePlacement.
    """
    b = pb.Chip()
    b.name = derived["name"]
    b.chip_path = derived["chip_path"]
    b.topology_path = derived["tile_config_path"]
    b.n_tiles = derived["n_tiles"]
    b.includes.extend(derived["includes"])

    sims = config["sims"]
    b.mill.verilator_config = sims["verilator"]
    b.mill.p2e_config = sims["p2e"]

    raw_cores = _raw_core_list(config["designs"])
    meta_cores = derived["cores"]
    if len(raw_cores) != len(meta_cores):
        raise ValueError(
            f"core count mismatch: config={len(raw_cores)} derived={len(meta_cores)}"
        )
    for raw, meta in zip(raw_cores, meta_cores):
        _fill_core(b.cores.add(), raw, meta, bbdir)

    tile_proto = _raw_tile_proto(config["designs"])
    for meta in derived["tiles"]:
        _fill_tile(b.tiles.add(), meta, tile_proto)

    targets = derived["targets"]
    for name, target in targets.items():
        p = b.profiles.add()
        p.name = name
        p.pkg = target["pkg"]
        p.compiler = target["compiler"]
        bank = target["bank"]
        p.bank_num = bank["num"]
        p.bank_width = bank["width"]
        p.bank_entries = bank["entries"]
        p.ball_ctest_dirs.extend(target["ball_ctest_dirs"])

    bemu = derived["bemu"]
    b.bemu.chip_main = bemu["chip_main"]
    b.bemu.tile_index = bemu["tile_index"]
    for ball in bemu["balls"]:
        e = b.bemu.balls.add()
        e.ball_class = ball["ball_class"]
        e.ball_dir = ball["ball_dir"]
        e.emu_lib = ball["emu_lib"]

    for key, val in derived["workload"]["cmake_param"].items():
        b.workload.cmake_param[key] = val
    return b


def write_chip_pb(
    config_path: Path,
    derived_path: Path,
    out_path: Path,
    bbdir: Path,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    derived = json.loads(derived_path.read_text(encoding="utf-8"))
    chip = fill_chip(config, derived, bbdir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(chip.SerializeToString())
    return out_path
