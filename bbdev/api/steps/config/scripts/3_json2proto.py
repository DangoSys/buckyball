#!/usr/bin/env python3
"""Build Chip protobuf from config.json + derived.json."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import chip_pb2 as pb  # noqa: E402
from ball_normalize import normalize_ball_domain


def _load_derive():
    path = Path(__file__).resolve().with_name("2_parameter_derivation.py")
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
        params = m["config"]["ball"]
        for key, value in params.items():
            e.ball_params[key] = str(value)
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


def _fill_rocket(msg: pb.RocketCpuConfig, d: dict[str, Any]) -> None:
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


def _fill_boom(msg: pb.BoomCpuConfig, d: dict[str, Any]) -> None:
    msg.fetch_width = d["fetchWidth"]
    msg.decode_width = d["decodeWidth"]
    msg.num_rob_entries = d["numRobEntries"]
    dc = d["dcache"]
    msg.dcache.n_sets = dc["nSets"]
    msg.dcache.n_ways = dc["nWays"]
    msg.dcache.n_mshrs = dc["nMSHRs"]
    ic = d["icache"]
    msg.icache.n_sets = ic["nSets"]
    msg.icache.n_ways = ic["nWays"]


def _check_cpu_kind(cpu: dict[str, Any], pkg: str) -> str:
    kind = cpu.get("kind")
    if kind not in ("rocket", "boom", "ant"):
        raise ValueError(f"{pkg}: kind must be 'rocket', 'boom' or 'ant', got {kind!r}")
    if not isinstance(cpu.get("config"), dict):
        raise ValueError(f"{pkg}: cpu.config must reference a TOML file")
    return kind


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


def _fill_rvv(msg: pb.RvvConfig, d: dict[str, Any], bbdir: Path) -> None:
    if type(d["enable"]) is not bool:
        raise ValueError("rvv.enable must be an explicit boolean")
    msg.enable = d["enable"]
    for key in ("laneNumber", "vLen", "eLen", "iBufWords", "memoryPorts"):
        if type(d[key]) is not int or d[key] <= 0:
            raise ValueError(f"rvv.{key} must be a positive integer")
    msg.source_path = _rel(bbdir, d["_file"])
    msg.lane_number = d["laneNumber"]
    msg.v_len = d["vLen"]
    msg.e_len = d["eLen"]
    msg.i_buf_words = d["iBufWords"]
    msg.memory_ports = d["memoryPorts"]


def _fill_tile_params(msg: pb.TileParamConfig, d: dict[str, Any]) -> None:
    msg.core_data_bytes = d["coreDataBytes"]
    msg.x_len = d["xLen"]
    msg.vaddr_bits = d["vaddrBits"]
    msg.paddr_bits = d["paddrBits"]
    msg.pg_idx_bits = d["pgIdxBits"]
    msg.pg_levels = d["pgLevels"]
    msg.n_pmps = d["nPMPs"]


def _fill_spm(msg: pb.SpmConfig, config: dict[str, Any]) -> None:
    msg.base = config["base"]
    msg.bytes = config["bytes"]
    msg.data_bits = config["dataBits"]


def _fill_cpu(msg: pb.CpuConfig, cpu: dict[str, Any], context: str, bbdir: Path) -> str:
    kind = _check_cpu_kind(cpu, context)
    msg.kind = kind
    msg.source_path = _rel(bbdir, cpu["_file"])
    if kind == "rocket":
        _fill_rocket(msg.rocket, cpu["config"])
    elif kind == "boom":
        _fill_boom(msg.boom, cpu["config"])
    else:
        config = cpu["config"]
        msg.ant.code_bytes = config["codeBytes"]
        msg.ant.task_bits = config["taskBits"]
        _fill_spm(msg.ant.tls, config["tls"])
    return kind


def _fill_core(ci: pb.CoreInstance, raw: dict[str, Any], meta: dict[str, Any], bbdir: Path) -> None:
    ci.index = meta["index"]
    ci.role = meta["role"]
    ci.pkg = meta["pkg"]
    ci.config_path = meta["config_path"]
    if raw["factory"] != meta["factory_class"]:
        raise ValueError("core factory differs between config and derived metadata")
    ci.factory_class = raw["factory"]
    ci.balldomain_base_dir = meta["balldomain_base_dir"]
    kind = _fill_cpu(ci.cpu, raw["cpu"], meta["pkg"], bbdir)
    domain = normalize_ball_domain(raw)
    if domain is not None:
        _fill_ball(ci.balldomain, domain, bbdir)
    if "memdomain" in raw:
        _fill_mem(ci.mem, raw["memdomain"], bbdir)
    if kind == "boom":
        bd = domain
        if isinstance(bd, dict) and bd.get("ballNum", 0):
            raise ValueError(f"{meta['pkg']}: kind=boom forbids balldomain.ballNum > 0")
    if "frontend" in raw:
        _fill_frontend(ci.frontend, raw["frontend"], bbdir)
    if "rvv" in raw:
        _fill_rvv(ci.rvv, raw["rvv"], bbdir)


def _fill_tile(tp: pb.TilePlacement, meta: dict[str, Any], proto: dict[str, Any]) -> None:
    tp.path = meta["path"]
    if meta["kind"] == "main":
        if "factory" in proto or meta["factory_class"]:
            raise ValueError("main tile must not specify a factory")
    else:
        if proto["factory"] != meta["factory_class"]:
            raise ValueError("tile factory differs between config and derived metadata")
        tp.factory_class = proto["factory"]
    tp.kind = pb.TILE_KIND_MAIN if meta["kind"] == "main" else pb.TILE_KIND_TILE
    tp.virtual_bank_count = meta["virtual_bank_count"]
    tp.core_indices.extend(meta["core_indices"])
    if "tss" in proto:
        _fill_spm(tp.tss, proto["tss"])
    if "controller_core_index" in meta:
        tp.controller_core_index = meta["controller_core_index"]
    tp.mem_ball_channel_num = meta["mem_ball_channel_num"]
    _fill_tile_params(tp.param, proto)
    sm = proto["sharedMem"]
    tp.shared_mem.enable = sm["enable"]
    tp.shared_mem.entries = sm["entries"]
    tp.shared_mem.bank_entries = sm["bankEntries"]
    tp.shared_mem.bank_width = sm["bankWidth"]
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

    Pair each tile's hardware parameters with its own derived placement.
    """
    b = pb.Chip()
    b.name = derived["name"]
    b.chip_path = derived["chip_path"]
    b.topology_path = derived["tile_config_path"]
    b.n_tiles = derived["n_tiles"]
    b.includes.extend(derived["includes"])

    sims = config["sims"]
    if "verilator" in sims:
        b.mill.verilator_config = sims["verilator"]
    if "p2e" in sims:
        b.mill.p2e_config = sims["p2e"]

    raw_cores = _raw_core_list(config["designs"])
    meta_cores = derived["cores"]
    if len(raw_cores) != len(meta_cores):
        raise ValueError(
            f"core count mismatch: config={len(raw_cores)} derived={len(meta_cores)}"
        )
    hart_ids = {hart["core_index"]: hart["hart_id"] for hart in derived["harts"]}
    for index, (raw, meta) in enumerate(zip(raw_cores, meta_cores)):
        core = b.cores.add()
        _fill_core(core, raw, meta, bbdir)
        if core.cpu.kind == "ant":
            core.ant_context_id = meta["ant_context_id"]
        else:
            core.hart_id = hart_ids[index]

    raw_tiles = _derive.iter_topology_tiles(config["designs"])
    for raw, meta in zip(raw_tiles, derived["tiles"], strict=True):
        _fill_tile(b.tiles.add(), meta, raw)

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
