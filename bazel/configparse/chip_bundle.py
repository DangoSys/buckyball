#!/usr/bin/env python3
"""Build ChipBundle protobuf from chip.toml + topology TOML. Only toml2json reads TOML."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tomllib
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from chip_bundle_pb2 import (  # type: ignore
    BallDomain,
    BallIdMapping,
    BallIsaEntry,
    BankConfig,
    BemuBall,
    BemuConfig,
    ChipBundle,
    CoreInstance,
    CoreParamConfig,
    DmaConfig,
    FrontendConfig,
    GpDomainConfig,
    MemAddrConfig,
    MemDomainConfig,
    MillConfig,
    MmioConfig,
    PrivateDCacheConfig,
    RocketBtbConfig,
    RocketCoreConfig,
    RocketDCacheConfig,
    RocketFpuConfig,
    RocketICacheConfig,
    RocketMulDivConfig,
    SharedMemConfig,
    TilePlacement,
    TlbConfig,
    TmaConfig,
    WorkloadConfig,
)
from chip_json import (
    _tile_cores,
    bank_params,
    bemu_paths,
    check_manifest,
    core_pkg,
    iter_cores,
    tile_files,
    unique_cores,
)
from toml2json import toml2json

_SYSTEM_FUNCTS = frozenset({0, 1, 16, 32, 33, 34, 35})


def _die(msg: str) -> None:
    print(f"chip_bundle: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _ball_dir(ball_class: str) -> str:
    if not ball_class.startswith("examples.balls."):
        _die(f"malformed ballClass: {ball_class!r}")
    directory = ball_class[len("examples.balls.") :].split(".", 1)[0]
    if not directory:
        _die(f"malformed ballClass: {ball_class!r}")
    return directory


def _parse_balldomain(data: dict) -> BallDomain:
    out = BallDomain()
    ball_num = data.get("ballNum")
    if isinstance(ball_num, int):
        out.ball_num = ball_num
    mappings = data.get("ballIdMappings")
    if not isinstance(mappings, list):
        return out
    for mapping in mappings:
        if not isinstance(mapping, dict):
            _die("ballIdMappings entry must be a table")
        ball_id = mapping.get("ballId")
        ball_name = mapping.get("ballName")
        ball_class = mapping.get("ballClass")
        config_path = mapping.get("config", "")
        in_bw = mapping.get("inBW", 0)
        out_bw = mapping.get("outBW", 0)
        if isinstance(config_path, dict):
            raw = config_path.get("_file")
            config_path = raw if isinstance(raw, str) else ""
        if not isinstance(ball_id, int) or ball_id < 0:
            _die(f"ballId must be a non-negative int: {mapping!r}")
        if not isinstance(ball_name, str) or not ball_name:
            _die(f"ballName must be a non-empty string: {mapping!r}")
        if not isinstance(ball_class, str) or not ball_class:
            _die(f"ballClass must be a non-empty string: {mapping!r}")
        if not isinstance(config_path, str):
            _die(f"config must be a string: {mapping!r}")
        if not isinstance(in_bw, int) or in_bw < 0:
            _die(f"inBW must be a non-negative int: {mapping!r}")
        if not isinstance(out_bw, int) or out_bw < 0:
            _die(f"outBW must be a non-negative int: {mapping!r}")
        out.mappings.append(
            BallIdMapping(
                ball_id=ball_id,
                ball_name=ball_name,
                ball_class=ball_class,
                ball_dir=_ball_dir(ball_class),
                config_path=config_path,
                in_bw=in_bw,
                out_bw=out_bw,
            )
        )
    isa = data.get("ballISA")
    if not isinstance(isa, list):
        return out
    for entry in isa:
        if not isinstance(entry, dict):
            _die("ballISA entry must be a table")
        mnemonic = entry.get("mnemonic")
        funct7 = entry.get("funct7")
        bid = entry.get("bid")
        if not isinstance(mnemonic, str) or not mnemonic:
            _die(f"ballISA mnemonic must be a non-empty string: {entry!r}")
        if not isinstance(funct7, int) or funct7 < 0 or funct7 >= 128:
            _die(f"ballISA funct7 must be in [0, 127]: {entry!r}")
        if not isinstance(bid, int) or bid < 0:
            _die(f"ballISA bid must be a non-negative int: {entry!r}")
        out.isa.append(BallIsaEntry(mnemonic=mnemonic, funct7=funct7, bid=bid))
    return out


def _require_table(obj: dict, key: str, ctx: str) -> dict:
    val = obj.get(key)
    if not isinstance(val, dict):
        _die(f"{ctx}: missing [{key}]")
    return val


def _require_int(val: object, name: str, ctx: str) -> int:
    if not isinstance(val, int):
        _die(f"{ctx}: {name} must be an int")
    return val


def _require_bool(val: object, name: str, ctx: str) -> bool:
    if not isinstance(val, bool):
        _die(f"{ctx}: {name} must be a bool")
    return val


def _nested_table(core: dict, key: str, pkg: str) -> dict:
    val = core.get(key)
    if isinstance(val, dict):
        return val
    _die(f"{pkg}: missing [{key}]")


def _balldomain_base_dir(core: dict, pkg: str) -> str:
    bd = core.get("balldomain")
    if isinstance(bd, dict):
        rel = bd.get("_file")
        if isinstance(rel, str):
            return str(Path(rel).parent)
    return f"examples/cores/{pkg}/configs/balldomains"


def _memdomain(repo: Path, pkg: str, core: dict) -> dict:
    mem = core.get("memdomain")
    if isinstance(mem, dict):
        return mem
    stub = repo / "examples" / "cores" / pkg / "configs" / "memdomains" / "default.toml"
    if not stub.is_file():
        _die(f"{pkg}: missing {stub}")
    return toml2json(stub, repo)


def _mem_domain_config(repo: Path, pkg: str, core: dict) -> MemDomainConfig:
    mem = _memdomain(repo, pkg, core)
    num, width, entries = bank_params(core, pkg, repo)
    bank_table = _require_table(mem, "bank", f"{pkg} memdomain")
    dma_table = _require_table(mem, "dma", f"{pkg} memdomain")
    tlb_table = _require_table(mem, "tlb", f"{pkg} memdomain")
    tma_table = _require_table(mem, "tma", f"{pkg} memdomain")
    mmio_table = _require_table(mem, "mmio", f"{pkg} memdomain")
    mem_table = _require_table(mem, "mem", f"{pkg} memdomain")
    bank = BankConfig(
        num=num,
        width=width,
        entries=entries,
        mask_len=_require_int(bank_table.get("maskLen"), "maskLen", pkg),
        channel=_require_int(bank_table.get("channel"), "channel", pkg),
    )
    dma = DmaConfig(
        n_xacts=_require_int(dma_table.get("nXacts"), "nXacts", pkg),
        burst_max_bytes=_require_int(
            dma_table.get("burstMaxBytes"), "burstMaxBytes", pkg
        ),
        bus_width=_require_int(dma_table.get("busWidth"), "busWidth", pkg),
        max_in_flight_mem_reqs=_require_int(
            dma_table.get("maxInFlightMemReqs"), "maxInFlightMemReqs", pkg
        ),
    )
    tlb = TlbConfig(size=_require_int(tlb_table.get("size"), "size", pkg))
    tma = TmaConfig(
        read_channel=_require_int(tma_table.get("readChannel"), "readChannel", pkg),
        write_channel=_require_int(tma_table.get("writeChannel"), "writeChannel", pkg),
    )
    mmio = MmioConfig(
        enable=_require_bool(mmio_table.get("enable"), "enable", f"{pkg} mmio"),
        bank_num=int(mmio_table.get("bankNum", 0)),
        bank_entries=int(mmio_table.get("bankEntries", 0)),
        bank_width=int(mmio_table.get("bankWidth", 0)),
        read_width=int(mmio_table.get("readWidth", 0)),
    )
    for field, name in (
        (mmio.bank_num, "bankNum"),
        (mmio.bank_entries, "bankEntries"),
        (mmio.bank_width, "bankWidth"),
        (mmio.read_width, "readWidth"),
    ):
        if field < 0:
            _die(f"{pkg}: mmio.{name} must be non-negative")
    mem_addr = MemAddrConfig(
        addr_len=_require_int(mem_table.get("addrLen"), "addrLen", pkg),
    )
    return MemDomainConfig(
        bank=bank,
        dma=dma,
        tlb=tlb,
        tma=tma,
        mmio=mmio,
        mem=mem_addr,
    )


def _parse_rocket_core(core: dict, pkg: str) -> RocketCoreConfig:
    t = _require_table(core, "rocketCore", pkg)
    mul = _require_table(t, "mulDiv", f"{pkg} rocketCore")
    fpu = _require_table(t, "fpu", f"{pkg} rocketCore")
    dcache = _require_table(t, "dcache", f"{pkg} rocketCore")
    icache = _require_table(t, "icache", f"{pkg} rocketCore")
    btb = _require_table(t, "btb", f"{pkg} rocketCore")
    return RocketCoreConfig(
        x_len=_require_int(t.get("xLen"), "xLen", pkg),
        pg_levels=_require_int(t.get("pgLevels"), "pgLevels", pkg),
        use_vm=_require_bool(t.get("useVM"), "useVM", pkg),
        use_zba=_require_bool(t.get("useZba"), "useZba", pkg),
        use_zbb=_require_bool(t.get("useZbb"), "useZbb", pkg),
        use_zbs=_require_bool(t.get("useZbs"), "useZbs", pkg),
        have_c_flush=_require_bool(t.get("haveCFlush"), "haveCFlush", pkg),
        mul_div=RocketMulDivConfig(
            enable=_require_bool(mul.get("enable"), "enable", f"{pkg} mulDiv"),
            mul_unroll=_require_int(mul.get("mulUnroll"), "mulUnroll", pkg),
            mul_early_out=_require_bool(mul.get("mulEarlyOut"), "mulEarlyOut", pkg),
            div_early_out=_require_bool(mul.get("divEarlyOut"), "divEarlyOut", pkg),
        ),
        fpu=RocketFpuConfig(
            enable=_require_bool(fpu.get("enable"), "enable", f"{pkg} fpu"),
            min_f_len=_require_int(fpu.get("minFLen"), "minFLen", pkg),
            f_len=_require_int(fpu.get("fLen"), "fLen", pkg),
        ),
        dcache=RocketDCacheConfig(
            n_sets=_require_int(dcache.get("nSets"), "nSets", pkg),
            n_ways=_require_int(dcache.get("nWays"), "nWays", pkg),
            n_mshrs=_require_int(dcache.get("nMSHRs"), "nMSHRs", pkg),
        ),
        icache=RocketICacheConfig(
            n_sets=_require_int(icache.get("nSets"), "nSets", pkg),
            n_ways=_require_int(icache.get("nWays"), "nWays", pkg),
        ),
        btb=RocketBtbConfig(
            enable=_require_bool(btb.get("enable"), "enable", f"{pkg} btb"),
            n_entries=_require_int(btb.get("nEntries"), "nEntries", pkg),
            n_ras=_require_int(btb.get("nRAS"), "nRAS", pkg),
        ),
    )


def _parse_frontend(core: dict, pkg: str) -> FrontendConfig:
    t = _nested_table(core, "frontend", pkg)
    return FrontendConfig(
        rob_entries=_require_int(t.get("robEntries"), "robEntries", pkg),
        rs_out_of_order_response=_require_bool(
            t.get("rsOutOfOrderResponse"), "rsOutOfOrderResponse", pkg
        ),
        bank_id_len=_require_int(t.get("bankIdLen"), "bankIdLen", pkg),
        vbank_id_upper_bound=_require_int(
            t.get("vbankIdUpperBound"), "vbankIdUpperBound", pkg
        ),
        shared_bank_id_base=_require_int(
            t.get("sharedBankIdBase"), "sharedBankIdBase", pkg
        ),
        iter_len=_require_int(t.get("iterLen"), "iterLen", pkg),
        sub_rob_enable=_require_bool(t.get("subRobEnable"), "subRobEnable", pkg),
        sub_rob_depth=_require_int(t.get("subRobDepth"), "subRobDepth", pkg),
    )


def _parse_gp_domain(core: dict, pkg: str) -> GpDomainConfig:
    t = _nested_table(core, "gpdomain", pkg)
    return GpDomainConfig(
        lane_number=_require_int(t.get("laneNumber"), "laneNumber", pkg),
        chaining_size=_require_int(t.get("chainingSize"), "chainingSize", pkg),
        v_len=_require_int(t.get("vLen"), "vLen", pkg),
        d_len=_require_int(t.get("dLen"), "dLen", pkg),
        e_len=_require_int(t.get("eLen"), "eLen", pkg),
        lane_scale=_require_int(t.get("laneScale"), "laneScale", pkg),
    )


def _parse_core_param(core: dict, pkg: str) -> CoreParamConfig:
    t = _nested_table(core, "core", pkg)
    return CoreParamConfig(
        core_data_bytes=_require_int(t.get("coreDataBytes"), "coreDataBytes", pkg),
        x_len=_require_int(t.get("xLen"), "xLen", pkg),
        vaddr_bits=_require_int(t.get("vaddrBits"), "vaddrBits", pkg),
        paddr_bits=_require_int(t.get("paddrBits"), "paddrBits", pkg),
        pg_idx_bits=_require_int(t.get("pgIdxBits"), "pgIdxBits", pkg),
        n_pmps=_require_int(t.get("nPMPs"), "nPMPs", pkg),
    )


def _parse_shared_mem(tile: dict) -> SharedMemConfig:
    t = _require_table(tile, "sharedMem", "tile")
    return SharedMemConfig(
        enable=_require_bool(t.get("enable"), "enable", "tile sharedMem"),
        entries=_require_int(t.get("entries"), "entries", "tile sharedMem"),
        input_channels=_require_int(
            t.get("inputChannels"), "inputChannels", "tile sharedMem"
        ),
        default_group_count=_require_int(
            t.get("defaultGroupCount"), "defaultGroupCount", "tile sharedMem"
        ),
    )


def _parse_private_dcache(tile: dict) -> PrivateDCacheConfig:
    t = tile.get("privateDCache")
    if not isinstance(t, dict):
        return PrivateDCacheConfig(enable=False)
    enable = t.get("enable")
    if not isinstance(enable, bool):
        _die("privateDCache.enable must be a bool")
    if not enable:
        return PrivateDCacheConfig(enable=False)
    return PrivateDCacheConfig(
        enable=True,
        ways=_require_int(t.get("ways"), "ways", "tile privateDCache"),
        capacity_kb=_require_int(
            t.get("capacityKB"), "capacityKB", "tile privateDCache"
        ),
        write_bytes=_require_int(
            t.get("writeBytes"), "writeBytes", "tile privateDCache"
        ),
        port_factor=_require_int(
            t.get("portFactor"), "portFactor", "tile privateDCache"
        ),
        mem_cycles=_require_int(t.get("memCycles"), "memCycles", "tile privateDCache"),
    )


def _core_instances(repo: Path, topo: dict) -> list[CoreInstance]:
    out: list[CoreInstance] = []
    for index, core in enumerate(iter_cores(topo)):
        rel = core.get("_file")
        if not isinstance(rel, str):
            _die("core config missing _file")
        pkg = core_pkg(rel)
        if not pkg:
            _die(f"unsupported core config path: {rel}")
        role = core.get("name")
        if role is not None and not isinstance(role, str):
            _die("core name must be a string")
        balldomain = core.get("balldomain")
        domain = (
            _parse_balldomain(balldomain)
            if isinstance(balldomain, dict)
            else BallDomain()
        )
        rocket = _parse_rocket_core(core, pkg)
        mem = _mem_domain_config(repo, pkg, core)
        inst = CoreInstance(
            index=index,
            role=role or "",
            pkg=pkg,
            config_path=rel,
            balldomain=domain,
            mem=mem,
            balldomain_base_dir=_balldomain_base_dir(core, pkg),
            rocket_core=rocket,
        )
        if domain.ball_num > 0:
            inst.frontend.CopyFrom(_parse_frontend(core, pkg))
            inst.gp_domain.CopyFrom(_parse_gp_domain(core, pkg))
            inst.core.CopyFrom(_parse_core_param(core, pkg))
        out.append(inst)
    return out


def _iter_topology_tiles(topo: dict) -> list[dict]:
    tiles = topo.get("tiles")
    if isinstance(tiles, list) and tiles:
        return [t for t in tiles if isinstance(t, dict)]
    template = topo.get("tileTemplate")
    if isinstance(template, dict):
        count = template.get("count")
        if not isinstance(count, int) or count < 1:
            _die("[tileTemplate].count must be a positive int")
        return [template] * count
    _die("topology must define [[tiles]] or [tileTemplate]")


def _tile_placements(topo: dict, cores: list[CoreInstance]) -> list[TilePlacement]:
    placements: list[TilePlacement] = []
    offset = 0
    for tile in _iter_topology_tiles(topo):
        tile_path = tile.get("_file")
        if not isinstance(tile_path, str):
            tile_path = ""
        n = len(_tile_cores(tile))
        shared = tile.get("sharedMem")
        vbc = 0
        if isinstance(shared, dict):
            raw = shared.get("virtualBankCount")
            if isinstance(raw, int) and raw > 0:
                vbc = raw
        if vbc == 0 and n > 0:
            vbc = max(cores[i].mem.bank.num for i in range(offset, offset + n))
        indices = list(range(offset, offset + n))
        has_buckyball = any(cores[i].balldomain.ball_num > 0 for i in indices)
        mem_ball_channel_num = 0
        if has_buckyball:
            raw = tile.get("memBallChannelNum")
            if not isinstance(raw, int):
                _die("tile with Buckyball cores must define memBallChannelNum")
            mem_ball_channel_num = raw
        placements.append(
            TilePlacement(
                path=tile_path,
                virtual_bank_count=vbc,
                core_indices=indices,
                mem_ball_channel_num=mem_ball_channel_num,
                private_dcache=_parse_private_dcache(tile),
                shared_mem=_parse_shared_mem(tile),
            )
        )
        offset += n
    return placements


def _n_tiles(topo: dict) -> int:
    top = topo.get("top")
    if not isinstance(top, dict):
        _die("topology missing [top]")
    n = top.get("nTiles")
    if not isinstance(n, int) or n < 1:
        _die("[top].nTiles must be a positive int")
    return n


def _bemu_balls(repo: Path, cores: list[CoreInstance]) -> list[BemuBall]:
    seen: set[str] = set()
    balls: list[BemuBall] = []
    for core in cores:
        domain = core.balldomain
        bid_to_class = {m.ball_id: m.ball_class for m in domain.mappings}
        for entry in domain.isa:
            if entry.funct7 in _SYSTEM_FUNCTS:
                continue
            ball_class = bid_to_class.get(entry.bid)
            if not ball_class:
                _die(
                    f"core {core.pkg}: ballISA funct7 {entry.funct7} "
                    f"references missing bid {entry.bid}"
                )
            if ball_class in seen:
                continue
            seen.add(ball_class)
            ball_dir = _ball_dir(ball_class)
            emu_lib = repo / "examples" / "balls" / ball_dir / "emu" / "src" / "lib.rs"
            if not emu_lib.is_file():
                _die(f"missing BEMU ball source for {ball_class}: {emu_lib}")
            balls.append(
                BemuBall(
                    ball_class=ball_class,
                    ball_dir=ball_dir,
                    emu_lib=str(emu_lib.resolve()),
                )
            )
    return balls


def _ball_ctest_dirs(repo: Path, core: CoreInstance) -> list[str]:
    dirs = []
    for mapping in core.balldomain.mappings:
        path = repo / "examples" / "balls" / mapping.ball_dir / "workloads" / "ctests"
        if not path.is_dir():
            _die(f"missing {path}")
        dirs.append(str(path))
    return dirs


def _lib_name(manifest: Path) -> str:
    with manifest.open("rb") as f:
        cargo = tomllib.load(f)
    lib = cargo.get("lib")
    if isinstance(lib, dict):
        name = lib.get("name")
        if isinstance(name, str) and name:
            return name
    pkg = cargo.get("package")
    if not isinstance(pkg, dict):
        _die(f"{manifest}: missing [package]")
    name = pkg.get("name")
    if not isinstance(name, str) or not name:
        _die(f"{manifest}: missing [package].name")
    return name.replace("-", "_")


def _rushb_defs(repo: Path, chip: str) -> dict[str, str]:
    manifest = repo / "examples" / "chips" / chip / "generated" / "bemu" / "Cargo.toml"
    if not manifest.is_file():
        _die(f"missing generated bemu manifest: {manifest}")
    lib = _lib_name(manifest)
    target = repo / "bebop" / "target" / chip / "release"
    return {
        "BUCKYBALL_RUSHB_BEMU_MANIFEST": str(manifest),
        "BUCKYBALL_RUSHB_BEMU_LIBRARY": str(target / f"lib{lib}.so"),
        "BUCKYBALL_RUSHB_VERILATOR_LIBRARY": str(
            target / "deps" / "libbebop_verilator.so"
        ),
    }


def _workload(
    repo: Path,
    chip: str,
    cores: list[CoreInstance],
    compiler_cores: list[str],
    chip_main: str,
) -> WorkloadConfig:
    defs: dict[str, str] = {
        "BUCKYBALL_WORKLOAD_CHIP": chip,
        "BUCKYBALL_CARGO_TARGET_DIR": str(repo / "bebop" / "target" / chip),
    }
    primary = compiler_cores[0] if len(compiler_cores) == 1 else ""
    if not primary:
        if chip != "poly":
            return WorkloadConfig(primary_core="", cmake_defs=defs)
        primary = "prefill"

    core = next(c for c in cores if c.pkg == primary)
    defs["BUCKYBALL_WORKLOAD_CORE"] = primary
    defs["BUCKYBALL_MLIR_BANK_NUM"] = str(core.mem.bank.num)
    defs["BUCKYBALL_MLIR_BANK_WIDTH_BITS"] = str(core.mem.bank.width)
    defs["BUCKYBALL_MLIR_BANK_DEPTH"] = str(core.mem.bank.entries)
    if len(compiler_cores) == 1:
        defs["BUCKYBALL_BALL_CTEST_DIRS"] = ";".join(_ball_ctest_dirs(repo, core))

    if chip == "poly":
        ids: dict[str, list[str]] = {}
        for inst in cores:
            if not inst.role:
                _die("poly core missing role")
            ids.setdefault(inst.role, []).append(str(inst.index))
        defs["BUCKYBALL_RUSHB_PLACEMENT_STRICT"] = "ON"
        for name, vals in ids.items():
            defs[f"_POLY_CORE_IDS_{name}"] = ";".join(vals)
        defs["BUCKYBALL_BALL_CTEST_DIRS"] = ";".join(_ball_ctest_dirs(repo, core))

    if chip_main:
        defs.update(_rushb_defs(repo, chip))

    return WorkloadConfig(primary_core=primary, cmake_defs=defs)


def build_bundle(repo: Path, chip: str, topology_rel: str | None = None) -> ChipBundle:
    repo = repo.resolve()
    chip_toml = repo / "examples" / "chips" / chip / "chip.toml"
    if not chip_toml.is_file():
        _die(f"missing {chip_toml}")
    manifest = toml2json(chip_toml, repo)
    check_manifest(manifest, chip)
    mill_table = manifest.get("chip")
    if not isinstance(mill_table, dict):
        _die(f"{chip}: missing [chip]")

    topo_rel = topology_rel or f"examples/chips/{chip}/configs/{chip}.toml"
    topo = toml2json(repo / topo_rel, repo)
    cores = _core_instances(repo, topo)
    compiler_cores = unique_cores(topo)
    chip_main, bemu_tile_index = bemu_paths(repo, chip, topo)
    includes = list(
        dict.fromkeys(manifest.get("includes", []) + topo.get("includes", []))
    )
    tiles = _tile_placements(topo, cores)
    n_tiles = _n_tiles(topo)
    if len(tiles) != n_tiles:
        _die(f"{chip}: [top].nTiles={n_tiles} but got {len(tiles)} tile(s)")

    bundle = ChipBundle(
        name=Path(topo_rel).stem,
        mill=MillConfig(
            verilator_config=str(mill_table.get("verilatorConfig", "")),
            p2e_config=str(mill_table.get("p2eConfig", "")),
        ),
        topology_path=topo_rel,
        includes=includes,
        n_tiles=n_tiles,
        tiles=tiles,
        cores=cores,
        compiler_cores=compiler_cores,
        bemu=BemuConfig(
            chip_main=chip_main,
            tile_index=bemu_tile_index,
            balls=_bemu_balls(repo, cores),
        ),
        workload=_workload(repo, chip, cores, compiler_cores, chip_main),
    )
    _ = tile_files(topo)
    return bundle


def chip_index(repo: Path, chip: str) -> dict:
    bundle = build_bundle(repo, chip)
    core = bundle.workload.primary_core
    if not core and len(bundle.compiler_cores) == 1:
        core = bundle.compiler_cores[0]
    return {
        "compilerCore": core,
        "topology": bundle.topology_path,
        "chipMain": bundle.bemu.chip_main,
        "bemuTileIndex": bundle.bemu.tile_index,
        "includes": list(bundle.includes),
    }


def _escape_rust(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _emit_dispatch(bundle: ChipBundle, core_name: str, out: Path) -> None:
    balls = list(bundle.bemu.balls)
    if not balls:
        out.write_text(
            "// AUTO-GENERATED — DO NOT EDIT\n"
            "use crate::inst::instruction::ExecContext;\n\n"
            "pub fn execute_known(\n"
            "    _ball_class: &str,\n"
            "    _funct: u32,\n"
            "    _xs1: u64,\n"
            "    _xs2: u64,\n"
            "    _ctx: &mut ExecContext,\n"
            ") -> u64 {\n"
            f'    panic!("no {core_name} BEMU ball implementation")\n'
            "}\n\n"
            "pub fn cycles_after_issue(_ball_class: &str, _funct: u32, _xs1: u64, _xs2: u64) -> u64 {\n"
            f'    panic!("no {core_name} BEMU ball latency implementation")\n'
            "}\n",
            encoding="utf-8",
        )
        return

    lines = [
        "// AUTO-GENERATED — DO NOT EDIT",
        "use crate::inst::instruction::ExecContext;",
        "",
    ]
    for ball in balls:
        lines.append(f'#[path = "{_escape_rust(ball.emu_lib)}"]')
        lines.append(f"mod {ball.ball_dir};")
        lines.append("")

    def chain(fn: str, panic_msg: str) -> list[str]:
        body: list[str] = []
        first = balls[0]
        body.append(f"    {first.ball_dir}::{fn}(ball_class, funct, xs1, xs2")
        if fn == "execute_known":
            body[-1] += ", ctx"
        body[-1] += ")"
        for ball in balls[1:]:
            body.append(
                "        .or_else(|| {dir}::{fn}(ball_class, funct, xs1, xs2{ctx}))".format(
                    dir=ball.ball_dir,
                    fn=fn,
                    ctx=", ctx" if fn == "execute_known" else "",
                )
            )
        body.append(f'        .unwrap_or_else(|| panic!("{panic_msg}"))')
        return body

    lines += [
        "pub fn execute_known(",
        "    ball_class: &str,",
        "    funct: u32,",
        "    xs1: u64,",
        "    xs2: u64,",
        "    ctx: &mut ExecContext,",
        ") -> u64 {",
    ]
    lines += chain(
        "execute_known",
        f"no {core_name} BEMU ball implementation for ballClass={{ball_class}} funct7={{funct}}",
    )
    lines += ["}", ""]
    lines += [
        "pub fn cycles_after_issue(ball_class: &str, funct: u32, xs1: u64, xs2: u64) -> u64 {",
    ]
    lines += chain(
        "cycles_after_issue",
        f"no {core_name} BEMU ball latency implementation for ballClass={{ball_class}} funct7={{funct}}",
    )
    lines += ["}", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


_BEMU_BUILD_RS = r"""#[path = "../../../../../bebop/src/nodes/bemu/build_support/mod.rs"]
mod build_support;

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let gen = manifest_dir
        .parent()
        .expect("bemu manifest must live under examples/chips/<chip>/generated/bemu");
    let dispatch = gen.join("dispatch.rs");
    let chip_json = gen.join("chip.json");
    if !dispatch.is_file() || !chip_json.is_file() {
        panic!(
            "missing generated bundle in {}; run: "
            "python3 bazel/configparse/chip_bundle.py --repo . --chip <chip> --all",
            gen.display()
        );
    }
    fs::copy(&dispatch, out_dir.join("chip_balls.rs")).expect("copy dispatch.rs");
    let json_path = chip_json.display().to_string().replace('\\', "/");
    fs::write(
        out_dir.join("bundle_embed.rs"),
        format!("pub const CHIP_BUNDLE_JSON: &str = include_str!(r\"{json_path}\");\n"),
    )
    .expect("write bundle_embed.rs");
    println!("cargo:rerun-if-changed={}", dispatch.display());
    println!("cargo:rerun-if-changed={}", chip_json.display());

    let engine = manifest_dir.join("../../../../../bebop/src/nodes/bemu");
    let native_dir = build_support::spike::native_dir(&engine);
    let spike_dir = native_dir.join("spike");
    let spike_install_dir = out_dir.join("spike_install");
    let spike_build_dir = out_dir.join("spike_build");
    build_support::spike::build_and_link(&native_dir, &spike_dir, &spike_build_dir, &spike_install_dir);
    build_support::rerun::emit_engine(&engine, &native_dir);
}
"""


def _emit_bemu_crate(repo: Path, chip: str, gen: Path) -> None:
    bemu = gen / "bemu"
    bemu.mkdir(parents=True, exist_ok=True)
    root = "../../../../../"
    cargo = f"""\
[package]
name = "bebop-bemu"
version = "0.1.0"
edition = "2021"

[lib]
path = "{root}bebop/src/nodes/bemu/src/lib.rs"
crate-type = ["rlib", "cdylib"]

[[bin]]
name = "bebop-bemu"
path = "{root}bebop/src/nodes/bemu/src/main.rs"

[[test]]
name = "test_bemu"
path = "{root}bebop/tests/test_bemu.rs"
harness = false

[dependencies]
snafu = "0.8"
once_cell = "1"
serde = {{ version = "1", features = ["derive"] }}
serde_json = "1"
bebop-elf = {{ path = "{root}bebop/src/nodes/lib/elf" }}
bebop-syscall = {{ path = "{root}bebop/src/nodes/lib/syscall" }}
bebop-dtb = {{ path = "{root}bebop/src/nodes/lib/dtb" }}
bebop-uart = {{ path = "{root}bebop/src/nodes/lib/uart" }}
bebop-bank-hash = {{ path = "{root}bebop/src/nodes/lib/bank-hash" }}
bebop-bemu-profile = {{ path = "{root}bebop/src/nodes/lib/bemu-profile" }}
bebop-rushb = {{ path = "{root}bebop/src/nodes/lib/rushB" }}

[build-dependencies]
cc = {{ version = "1", features = ["parallel"] }}
toml = "0.8"

[dev-dependencies]
libtest-mimic = "0.8"
assert_cmd = "2"
walkdir = "2"
predicates = "3"
chrono = {{ version = "0.4", default-features = false, features = ["clock"] }}
clap = {{ version = "4", features = ["derive"] }}
toml = "0.8"

[features]
default = ["bemu"]
bemu = []
difftest = []
"""
    (bemu / "Cargo.toml").write_text(cargo, encoding="utf-8")
    (bemu / "build.rs").write_text(_BEMU_BUILD_RS, encoding="utf-8")
    _emit_bebop_shim(repo, gen)


def _emit_bebop_shim(repo: Path, gen: Path) -> None:
    root = "../../../../../"
    shim = gen / "bebop"
    shim.mkdir(parents=True, exist_ok=True)
    cargo = f"""\
[package]
name = "bebop"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "bebop"
path = "{root}bebop/src/main.rs"

[dependencies]
bebop-verilator = {{ path = "{root}bebop/src/nodes/verilator" }}
bebop-bemu = {{ path = "../bemu" }}
bebop-dasm = {{ path = "{root}bebop/src/nodes/lib/dasm" }}
bebop-bank-hash = {{ path = "{root}bebop/src/nodes/lib/bank-hash" }}
bebop-bemu-profile = {{ path = "{root}bebop/src/nodes/lib/bemu-profile" }}
bebop-fd-redirect = {{ path = "{root}bebop/src/nodes/lib/fd-redirect" }}
bebop-rtl-trace = {{ path = "{root}bebop/src/nodes/lib/rtl-trace" }}
bebop-uart = {{ path = "{root}bebop/src/nodes/lib/uart" }}
clap = {{ version = "4", features = ["derive"] }}
libc = "0.2"
log = "0.4"
env_logger = "0.11"
nix = {{ version = "0.29", features = ["fs", "mman", "signal", "process"] }}
toml = "0.8"
ctrlc = "3"
camino = "1.1"
snafu = "0.8"
serde = {{ version = "1", features = ["derive"] }}
serde_json = "1"
duct = "0.13"

[features]
default = []
verilator = ["dep:bebop-verilator"]
bemu = ["dep:bebop-bemu"]
difftest = ["bemu", "bebop-bemu/difftest"]
"""
    (shim / "Cargo.toml").write_text(cargo, encoding="utf-8")
    build_rs = (repo / "bebop" / "build.rs").read_text(encoding="utf-8")
    (shim / "build.rs").write_text(build_rs, encoding="utf-8")
    _ = repo


def install_bundle(
    repo: Path, chip: str, bundle: ChipBundle, pb: Path, text: Path | None = None
) -> Path:
    repo = repo.resolve()
    gen = repo / "examples" / "chips" / chip / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pb.resolve(), gen / "chip.pb")
    if text is not None:
        shutil.copy2(text.resolve(), gen / "chip.textproto")
    from google.protobuf.json_format import MessageToJson

    (gen / "chip.json").write_text(MessageToJson(bundle), encoding="utf-8")
    _emit_dispatch(bundle, f"bemu-{chip}", gen / "dispatch.rs")
    _emit_bemu_crate(repo, chip, gen)
    return gen / "bemu" / "Cargo.toml"


def write_bundle(
    repo: Path,
    chip: str,
    out_pb: Path,
    out_text: Path,
    topology_rel: str | None = None,
) -> ChipBundle:
    from google.protobuf import text_format

    bundle = build_bundle(repo, chip, topology_rel=topology_rel)
    header = "# AUTO-GENERATED — DO NOT EDIT\n"
    out_pb.parent.mkdir(parents=True, exist_ok=True)
    out_text.parent.mkdir(parents=True, exist_ok=True)
    out_pb.write_bytes(bundle.SerializeToString())
    out_text.write_text(
        header + text_format.MessageToString(bundle, as_utf8=True),
        encoding="utf-8",
    )
    return bundle


def write_chip_bundles(repo: Path, chip: str) -> list[Path]:
    configs = repo / "examples" / "chips" / chip / "configs"
    if not configs.is_dir():
        _die(f"missing {configs}")
    gen = repo / "examples" / "chips" / chip / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for path in sorted(configs.glob("*.toml")):
        rel = path.relative_to(repo).as_posix()
        stem = path.stem
        pb = gen / f"{stem}.pb"
        text = gen / f"{stem}.textproto"
        bundle = write_bundle(repo, chip, pb, text, topology_rel=rel)
        out.append(pb)
        if stem == chip:
            install_bundle(repo, chip, bundle, pb, text)
    if not out:
        _die(f"{chip}: no topology under {configs}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--chip", required=True)
    parser.add_argument("--topology")
    parser.add_argument("--out-pb", type=Path)
    parser.add_argument("--out-text", type=Path)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--stamp", type=Path)
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not repo.is_dir():
        _die(f"repo is not a directory: {repo}")

    if args.all:
        write_chip_bundles(repo, args.chip)
        return

    if not args.out_pb or not args.out_text:
        _die("pass --out-pb and --out-text, or --all")
    topo = args.topology
    if topo and not topo.startswith("examples/"):
        topo = f"examples/chips/{args.chip}/configs/{topo}"
    bundle = write_bundle(
        repo, args.chip, args.out_pb, args.out_text, topology_rel=topo
    )
    if args.install:
        install_bundle(repo, args.chip, bundle, args.out_pb, args.out_text)
    if args.stamp:
        args.stamp.parent.mkdir(parents=True, exist_ok=True)
        args.stamp.write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
