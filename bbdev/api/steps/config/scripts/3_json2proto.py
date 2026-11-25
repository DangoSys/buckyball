#!/usr/bin/env python3
"""Build Design protobuf from config.json + derived.json."""

from __future__ import annotations

import importlib.util
import json
import hashlib
import re
import tomllib
from google.protobuf.json_format import ParseDict
import sys
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import design_pb2 as pb  # noqa: E402
from ball_normalize import normalize_ball_domain
from ball_parameters import pack_parameters


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


def _raw_thread_list(designs: dict[str, Any]) -> list[dict[str, Any]]:
    return list(_derive.iter_threads(designs))


def _fill_bank(msg: pb.BankConfig, d: dict[str, Any]) -> None:
    msg.num = d["num"]
    msg.width = d["width"]
    msg.entries = d["entries"]
    msg.mask_len = d["maskLen"]
    msg.request_ports = d["requestPorts"]
    msg.layout_enable = d["layoutEnable"]


def _fill_mem(msg: pb.MemDomainConfig, d: dict[str, Any], bbdir: Path) -> None:
    msg.source_path = _rel(bbdir, d["_file"])
    _fill_bank(msg.bank, d["bank"])
    dma = d["dma"]
    msg.dma.bus_width = dma["busWidth"]
    mmio = d["mmio"]
    msg.mmio.enable = mmio["enable"]
    msg.mmio.bank_num = mmio["bankNum"]
    msg.mmio.bank_entries = mmio["bankEntries"]
    msg.mmio.bank_width = mmio["bankWidth"]
    msg.mmio.read_width = mmio["readWidth"]
    msg.addr_len = d["addrLen"]


def _fill_ball(msg: pb.BallDomain, d: dict[str, Any], bbdir: Path) -> None:
    # A declared BallDomain is present in design.pb even before it has Balls.
    msg.SetInParent()
    for ball in d["balls"]:
        instance = msg.balls.add()
        instance.id = ball["id"]
        instance.name = ball["name"]
        instance.component = ball["component"]
        root = _derive.ball_root(ball)
        instance.source_path = _rel(bbdir, root)
        instance.rtl_class = ".".join(_derive.ball_class(root))
        port = ball["port"]
        instance.in_bw = port["inBW"]
        instance.out_bw = port["outBW"]
        instance.mmio_read_bw = port["mmioReadBW"]
        instance.mmio_write_bw = port["mmioWriteBW"]
        # A Ball's default config also holds its interface and its host alone.
        pack_parameters(
            instance,
            {
                key: value
                for key, value in ball["param"].items()
                if key not in ("_file", "ball", "host")
            },
        )
        for instruction in ball["instructions"]:
            entry = instance.instructions.add()
            entry.mnemonic = instruction["mnemonic"]
            entry.funct7 = instruction["funct7"]


def _check_cpu_kind(cpu: dict[str, Any], pkg: str) -> str:
    kind = cpu.get("kind")
    if kind != "ant":
        raise ValueError(f"{pkg}: kind must be 'ant', got {kind!r}")
    if not isinstance(cpu.get("config"), dict):
        raise ValueError(f"{pkg}: cpu.config must reference a TOML file")
    return kind


def _fill_frontend(msg: pb.FrontendConfig, d: dict[str, Any], bbdir: Path) -> None:
    msg.source_path = _rel(bbdir, d["_file"])
    msg.rob_entries = d["robEntries"]
    msg.rs_out_of_order_response = d["rsOutOfOrderResponse"]
    msg.bank_id_len = d["bankIdLen"]
    msg.max_private_vbank = d["maxPrivateVbank"]
    msg.shared_vbank_base = d["sharedVbankBase"]
    msg.iter_len = d["iterLen"]
    msg.sub_rob_enable = d["subRobEnable"]
    msg.sub_rob_depth = d["subRobDepth"]
    msg.bank_alias_enable = d["bankAliasEnable"]


def _fill_gpdomain(msg: pb.GpDomainConfig, d: dict[str, Any], bbdir: Path) -> None:
    if type(d["enable"]) is not bool:
        raise ValueError("gpdomain.enable must be an explicit boolean")
    msg.enable = d["enable"]
    if type(d["baseInteger"]) is not bool:
        raise ValueError("gpdomain.baseInteger must be an explicit boolean")
    msg.base_integer = d["baseInteger"]
    if type(d["fp64"]) is not bool:
        raise ValueError("gpdomain.fp64 must be an explicit boolean")
    msg.fp64 = d["fp64"]
    for key in ("laneNumber", "vLen", "eLen", "iBufWords", "memoryPorts"):
        if type(d[key]) is not int or d[key] <= 0:
            raise ValueError(f"gpdomain.{key} must be a positive integer")
    msg.source_path = _rel(bbdir, d["_file"])
    msg.lane_number = d["laneNumber"]
    msg.v_len = d["vLen"]
    msg.e_len = d["eLen"]
    msg.i_buf_words = d["iBufWords"]
    msg.memory_ports = d["memoryPorts"]


def _fill_hart_params(msg: pb.HartParamConfig, d: dict[str, Any]) -> None:
    msg.thread_data_bytes = d["threadDataBytes"]
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
    config = cpu["config"]
    msg.ant.code_bytes = config["codeBytes"]
    msg.ant.task_bits = config["taskBits"]
    _fill_spm(msg.ant.tls, config["tls"])
    return kind


def _hardware_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _hardware_fields(item) for key, item in value.items() if key != "_file"
        }
    if isinstance(value, list):
        return [_hardware_fields(item) for item in value]
    return value


def _thread_signature(thread: pb.ThreadInstance) -> int:
    hardware = pb.ThreadInstance()
    hardware.CopyFrom(thread)
    for ball in hardware.balldomain.balls:
        ball.ClearField("rtl_class")
        ball.ClearField("source_path")
    for field in (
        "index",
        "role",
        "config_path",
        "source_path",
        "rtl_class",
        "balldomain_base_dir",
        "local_id",
        "signature",
        "transport",
        "profile",
    ):
        hardware.ClearField(field)
    for message in (hardware.cpu, hardware.mem, hardware.frontend, hardware.gpdomain):
        message.ClearField("source_path")
    digest = hashlib.sha256(hardware.SerializeToString(deterministic=True)).digest()
    return int.from_bytes(digest[:8], "little")


def _fill_thread(
    ci: pb.ThreadInstance,
    raw: dict[str, Any],
    meta: dict[str, Any],
    bbdir: Path,
    *,
    hardware: bool,
) -> None:
    ci.index = meta["index"]
    ci.local_id = meta["thread_id"]
    ci.role = meta["role"]
    ci.pkg = meta["pkg"]
    ci.profile = raw["compiler"]["name"] if "compiler" in raw else meta["role"]
    ci.config_path = meta["config_path"]
    source = _derive.component_root(raw["_file"])[0]
    ci.source_path = _rel(bbdir, source)
    if hardware:
        classes = []
        for implementation in sorted((source / "arch").rglob("*.scala")):
            text = implementation.read_text()
            names = re.findall(
                r"class (\w+)\([^{}]*?\)\s+extends Module\s+with HasTLink\b", text, re.S
            )
            if names:
                package = re.search(r"(?m)^package ([\w.]+)", text).group(1)
                classes.extend(f"{package}.{name}" for name in names)
        if len(classes) != 1:
            raise ValueError(
                f"{source}: expected one Thread implementation, found {classes}"
            )
        ci.rtl_class = classes[0]
    ci.balldomain_base_dir = meta["balldomain_base_dir"]
    if "cpu" in raw:
        _fill_cpu(ci.cpu, raw["cpu"], meta["pkg"], bbdir)
    domain = normalize_ball_domain(raw)
    if domain is not None:
        _fill_ball(ci.balldomain, domain, bbdir)
    if "memdomain" in raw:
        _fill_mem(ci.mem, raw["memdomain"], bbdir)
    if "frontend" in raw:
        _fill_frontend(ci.frontend, raw["frontend"], bbdir)
    if "gpdomain" in raw:
        _fill_gpdomain(ci.gpdomain, raw["gpdomain"], bbdir)
    if "transport" in raw:
        ParseDict(_hardware_fields(raw["transport"]), ci.transport)
    ci.signature = _thread_signature(ci)


def fill_design(
    config: dict[str, Any], derived: dict[str, Any], bbdir: Path
) -> pb.Design:
    """Serialize explicitly declared instances and their typed parameters into design.pb."""
    b = pb.Design()
    b.name = derived["name"]
    b.path = derived["path"]
    b.n_harts = derived["n_harts"]
    b.includes.extend(derived["includes"])

    for key in ("ddr", "clint", "plic", "scu", "trace"):
        ParseDict(_hardware_fields(config["device"][key]), getattr(b.device, key))
    bootrom = config["device"]["bootrom"]
    image_path = bbdir / bootrom["image"]
    image = bytearray(image_path.read_bytes())
    if len(image) < 24 or len(image) > bootrom["bytes"]:
        raise ValueError(
            "BootROM image must contain its configuration header and fit the ROM"
        )
    if any(image[8:24]):
        raise ValueError(
            "BootROM image configuration header must contain zero placeholders"
        )
    image[8:16] = b.n_harts.to_bytes(8, "little")
    image[16:24] = bootrom["entry"].to_bytes(8, "little")
    b.device.bootrom.base = bootrom["base"]
    b.device.bootrom.bytes = bootrom["bytes"]
    b.device.bootrom.image.extend(byte if byte < 128 else byte - 256 for byte in image)
    b.includes.append(_rel(bbdir, image_path))

    ParseDict(
        {
            key: _hardware_fields(config["sims"][key])
            for key in ("diffTest", "memory", "verilatorThreads")
        },
        b.sims,
    )
    if not b.device.IsInitialized() or not b.sims.IsInitialized():
        raise ValueError("Device and simulation parameters must be complete")
    if b.sims.verilator_threads == 0:
        raise ValueError("sims.verilatorThreads must be positive")

    # The device directory holds the simulation entries; the design directory
    # above it holds the output. A hart alone, configured by <hart>/configs/
    # default.toml, holds its entry and its output itself.
    root = Path(derived["path"]).parent.parent
    owner = root if root.parent.name == "harts" else root.parent.parent
    ((name, registration),) = config["sims"]["targets"].items()
    target = b.simulation
    target.selection = config["sims"]["selection"]
    target.output = str(owner / "output/sims" / target.selection)
    # A hart entry is under its hart class: harts/<class>/sims/hart.
    if name == "device":
        entry = root / "sims" / name
    elif name == "hart":
        entry = (
            root.parent.parent
            / "harts"
            / target.selection.split("/")[1]
            / "sims"
            / name
        )
    else:
        raise ValueError(f"invalid simulation target: {name}")
    target.directory = str(entry)
    for field in ("emu", "stack", "workloads", "regression"):
        setattr(target, field, str(entry / field))
    target.arch.directory = str(entry / "arch")
    target.arch.module = "simulation"
    target.arch.config = str(Path(target.output) / "generated/design.pb")
    target.entry_class = registration["config"]
    target.top = "BBSimHarness"
    manifest = bbdir / target.emu / "Cargo.toml"
    target.emu_package = tomllib.loads(manifest.read_text())["package"]["name"]
    for entry in (
        Path(target.arch.directory) / "build.sc",
        Path(target.stack) / "CMakeLists.txt",
        Path(target.workloads) / "CMakeLists.txt",
    ):
        if not (bbdir / entry).is_file():
            raise FileNotFoundError(bbdir / entry)
    from verification import ip_registrations

    entries = config["sims"]["verification"]["registry"]["ips"]
    # An IP listed by name alone is present but has no runnable verification profile.
    profiles = ip_registrations(
        bbdir, [entry for entry in entries if set(entry) != {"ip"}]
    )
    b.verification.ips.extend(dict.fromkeys(entry["ip"] for entry in entries))
    b.verification.profiles.extend(profiles)

    for raw in _derive.configured_harts(config["device"]):
        _append_hart(b, raw, bbdir)

    if "sharedBankFabric" in config["device"]:
        fabric = config["device"]["sharedBankFabric"]
        harts = {hart.hart_id for hart in b.device.harts}
        owner, participants = fabric["ownerHart"], fabric["participantHarts"]
        if (
            type(owner) is not int
            or owner not in harts
            or not participants
            or owner in participants
            or len(set(participants)) != len(participants)
            or set(participants) | {owner} != harts
            or len(fabric["bankNodes"]) != len(harts)
        ):
            raise ValueError(
                "sharedBankFabric must explicitly connect each declared hart once"
            )
        ParseDict(_hardware_fields(fabric), b.device.shared_bank_fabric)

    domains = [
        hart.accelerator.balldomain
        for hart in b.device.harts
        if hart.HasField("accelerator")
    ]
    domains.extend(thread.balldomain for thread in b.threads)
    b.verification.balls.extend(
        dict.fromkeys(ball.component for domain in domains for ball in domain.balls)
    )

    targets = derived["targets"]
    compiled = {}
    for profile in _derive.compiler_profiles(config["device"]):
        name = profile["name"]
        raw = profile["config"]
        meta = {
            "index": 0,
            "thread_id": 0,
            "role": name,
            "pkg": profile["component"],
            "config_path": _rel(bbdir, raw["_file"]),
            "balldomain_base_dir": _derive._balldomain_base_dir(
                bbdir, raw, profile["component"]
            ),
        }
        instance = pb.ThreadInstance()
        _fill_thread(instance, raw, meta, bbdir, hardware=False)
        if name in compiled:
            if compiled[name].thread.signature != instance.signature:
                raise ValueError(
                    f"compiler {name}: instances have different parameters; declare distinct compiler names"
                )
            continue
        p = b.profiles.add()
        compiled[name] = p
        p.name = name
        p.ball_ctest_dirs.extend(targets[name]["ball_ctest_dirs"])
        p.thread.CopyFrom(instance)
        # Programs of the profile run on the first hart that runs it, on its
        # accelerator or on one of its threads.
        p.hart.CopyFrom(
            next(
                hart
                for hart in b.device.harts
                if hart.profile == name
                or any(b.threads[i].profile == name for i in hart.thread_indices)
            )
        )

    bemu = derived["bemu"]
    for ball in bemu["balls"]:
        e = b.bemu.balls.add()
        e.emu_class = ball["emu_class"]
        e.component = ball["component"]
        e.emu_lib = ball["emu_lib"]

    harts = {hart.hart_id for hart in b.device.harts}
    tests = config["workloads"]
    for key, field in (("controlHart", "control_hart"), ("reportHart", "report_hart")):
        hart = tests[key]
        if type(hart) is not int or hart not in harts:
            raise ValueError(f"workload {key}: unknown hart {hart!r}")
        setattr(b.workload, field, hart)
    return b


def write_thread_pb(raw: dict[str, Any], out_path: Path, bbdir: Path, pkg: str) -> Path:
    """One thread config, independent of any design: the compiler input."""
    rel = _rel(bbdir, raw["_file"])
    meta = {
        "index": 0,
        "thread_id": 0,
        "role": "",
        "pkg": pkg,
        "config_path": rel,
        "balldomain_base_dir": _derive._balldomain_base_dir(bbdir, raw, pkg),
    }
    thread = pb.ThreadInstance()
    _fill_thread(thread, raw, meta, bbdir, hardware=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(thread.SerializeToString())
    return out_path


def _append_hart(design: pb.Design, raw: dict[str, Any], bbdir: Path) -> None:
    hart = design.device.harts.add()
    hart.path = _rel(bbdir, raw["_file"])
    component = raw.get("component")
    if not isinstance(component, str) or not component:
        raise ValueError("Hart component must be explicitly named")
    hart.component = component
    hart.profile = raw["compiler"]["name"] if "compiler" in raw else ""
    hart.hart_id = raw["hartId"]
    if "nodeId" in raw:
        hart.node_id = raw["nodeId"]
    hart.reset_vector = raw["resetVector"]
    hart.mtime_address = raw["mtimeAddress"]
    hart.service = raw["service"]
    # The hart's stack mirrors where its config lives: <root>/configs/<sub>
    # has its stack at <root>/stack/<sub>.
    root, sub = _derive.component_root(raw["_file"])
    stack = root / "stack" / sub
    if not (stack / "CMakeLists.txt").is_file():
        raise FileNotFoundError(f"missing hart stack project: {stack}")
    hart.stack_path = _rel(bbdir, stack)
    for key in ("memory", "fetch", "fpu", "interconnect", "controller"):
        ParseDict(_hardware_fields(raw[key]), getattr(hart, key))
    if "tbus" in raw:
        ParseDict(_hardware_fields(raw["tbus"]), hart.tbus)
    if not hart.IsInitialized():
        raise ValueError("Hart IP configuration is incomplete")
    if "implementation" in raw:
        from importlib import import_module

        implementation = import_module(
            f"{raw['implementation']['component']}_params_pb2"
        ).Params()
        ParseDict(
            _hardware_fields(
                {k: v for k, v in raw["implementation"].items() if k != "component"}
            ),
            implementation,
        )
        hart.implementation.Pack(implementation, deterministic=True)
    if "param" in raw or "implementation" not in raw:
        _fill_hart_params(hart.param, raw["param"])
    if "tss" in raw:
        _fill_spm(hart.tss, raw["tss"])
    shared = raw["sharedMem"]
    hart.virtual_bank_count = shared["virtualBankCount"]
    hart.shared_mem.enable = shared["enable"]
    hart.shared_mem.entries = shared["entries"]
    hart.shared_mem.bank_entries = shared["bankEntries"]
    hart.shared_mem.bank_width = shared["bankWidth"]
    hart.shared_mem.input_channels = shared["inputChannels"]
    hart.shared_mem.default_group_count = shared["defaultGroupCount"]
    hart.shared_mem.virtual_bank_count = shared["virtualBankCount"]
    hart.shared_mem.latency = shared["latency"]
    if "dataLinks" in raw:
        for link in raw["dataLinks"]:
            topology = hart.data_links.add()
            topology.threads.extend(link["threads"])
            topology.banks = link["banks"]
    if "accelerator" in raw:
        acc = raw["accelerator"]
        _fill_ball(hart.accelerator.balldomain, acc["balldomain"], bbdir)
        _fill_mem(hart.accelerator.mem, acc["memdomain"], bbdir)
        _fill_frontend(hart.accelerator.frontend, acc["frontend"], bbdir)
        _fill_gpdomain(hart.accelerator.gpdomain, acc["gpdomain"], bbdir)
        ParseDict(_hardware_fields(acc["transport"]), hart.accelerator.transport)
    threads = raw["threads"]
    ids = [thread["id"] for thread in threads]
    if any(type(value) is not int or value < 0 for value in ids) or len(
        set(ids)
    ) != len(ids):
        raise ValueError(
            "Hart-local thread IDs must be explicit, non-negative and unique"
        )
    offset = len(design.threads)
    for local, thread in enumerate(threads):
        index = offset + local
        package = thread["component"]
        if not isinstance(package, str) or not package.isidentifier():
            raise ValueError("invalid thread component")
        meta = {
            "index": index,
            "thread_id": thread["id"],
            "role": thread["name"],
            "pkg": package,
            "config_path": _rel(bbdir, thread["_file"]),
            "balldomain_base_dir": _derive._balldomain_base_dir(bbdir, thread, package),
        }
        item = design.threads.add()
        _fill_thread(item, thread, meta, bbdir, hardware=True)
        hart.thread_indices.append(index)
    # The signature names the kind of hart software runs on; where an instance
    # sits (hart, config file, NoC nodes) is not part of it.
    hardware = pb.HartPlacement()
    hardware.CopyFrom(hart)
    for field in ("hart_id", "path", "thread_indices", "stack_path"):
        hardware.ClearField(field)
    for ball in hardware.accelerator.balldomain.balls:
        ball.ClearField("source_path")

    def clear_nodes(message):
        for field, value in message.ListFields():
            if field.name == "node_id":
                message.node_id = 0
            elif field.message_type is not None:
                for item in [value] if hasattr(value, "ListFields") else value:
                    clear_nodes(item)

    clear_nodes(hardware)
    payload = hardware.SerializeToString(deterministic=True) + b"".join(
        design.threads[index].signature.to_bytes(8, "little")
        for index in hart.thread_indices
    )
    hart.signature = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def write_hart_pb(raw: dict[str, Any], out_path: Path, bbdir: Path) -> Path:
    design = pb.Design(name=f"hart-{raw['hartId']}", n_harts=1)
    _append_hart(design, raw, bbdir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(design.SerializeToString(deterministic=True))
    return out_path


def fill_ball_design(config: dict[str, Any], bbdir: Path) -> pb.Design:
    """design.pb of a Ball alone: its host and the one Ball, no device."""
    b = pb.Design()
    ((ball,),) = [config["ball"]["balldomain"]["balls"]]
    b.name = ball["component"]
    b.includes.extend(_rel(bbdir, path) for path in config["includes"])
    host = config["ball"]["host"]
    h = b.ball.host
    h.rob_entries = host["robEntries"]
    h.iter_bits = host["iterBits"]
    h.bank_id_bits = host["bankIdBits"]
    h.data_bits = host["dataBits"]
    h.mem_addr_bits = host["memAddrBits"]
    h.bank_alias_enable = host["bankAliasEnable"]
    h.operators = host["operators"]
    h.bank_num = host["bank"]["num"]
    h.bank_entries = host["bank"]["entries"]
    h.bank_width = host["bank"]["width"]
    h.bank_mask_len = host["bank"]["maskLen"]
    h.bank_request_ports = host["bank"]["requestPorts"]
    h.mmio_enable = host["mmio"]["enable"]
    h.mmio_bank_num = host["mmio"]["bankNum"]
    h.mmio_bank_entries = host["mmio"]["bankEntries"]
    h.mmio_bank_width = host["mmio"]["bankWidth"]
    h.mmio_read_width = host["mmio"]["readWidth"]
    h.sub_rob_enable = host["subRob"]["enable"]
    h.sub_rob_depth = host["subRob"]["depth"]
    h.pmp_address_bits = host["pmp"]["addressBits"]
    h.pmp_entries = host["pmp"]["entries"]
    _fill_ball(b.ball.balldomain, config["ball"]["balldomain"], bbdir)

    # The entry of a Ball alone, whose classpath holds the Ball: the prefab
    # one for a prefab Ball, its design's device's for a design's Ball,
    # examples/designs/<design>/balls/<ball>. Its output is under the Ball's
    # own directory.
    root = _derive.ball_root(ball)
    if root.is_relative_to(bbdir / "examples/prefab/balls"):
        entry = Path("examples/prefab/sims/ball")
    else:
        design = root.relative_to(bbdir / "examples/designs").parts[0]
        (device,) = (bbdir / "examples/designs" / design / "devices").iterdir()
        entry = (
            Path("examples/designs") / design / "devices" / device.name / "sims/ball"
        )
    target = b.simulation
    target.selection = config["sims"]["selection"]
    target.output = str(Path(_rel(bbdir, root)) / "output/sims" / target.selection)
    target.directory = str(entry)
    target.arch.directory = str(entry / "arch")
    target.arch.module = "simulation"
    target.arch.config = str(Path(target.output) / "generated/design.pb")
    target.entry_class = "framework.ball.sims.Elaborate"
    target.top = "BallDomain"
    if not (bbdir / target.arch.directory / "build.sc").is_file():
        raise FileNotFoundError(bbdir / target.arch.directory / "build.sc")
    return b


def fill_thread_design(config: dict[str, Any], bbdir: Path) -> pb.Design:
    """design.pb of a thread alone: its host and the one thread, no device."""
    b = pb.Design()
    raw = config["thread"]["config"]
    root = _derive.component_root(raw["_file"])[0]
    pkg = root.name
    b.name = pkg
    b.includes.extend(_rel(bbdir, path) for path in config["includes"])
    host = config["thread"]["host"]
    h = b.thread.host
    h.x_len = host["xLen"]
    h.virtual_bank_count = host["virtualBankCount"]
    _fill_spm(h.tss, host["tss"])
    h.pmp_address_bits = host["pmp"]["addressBits"]
    h.pmp_entries = host["pmp"]["entries"]
    h.memory_base = host["memory"]["base"]
    h.memory_bytes = host["memory"]["bytes"]
    ParseDict(
        {
            key: _hardware_fields(host["sims"][key])
            for key in ("diffTest", "memory", "verilatorThreads")
        },
        b.sims,
    )
    if b.sims.verilator_threads == 0:
        raise ValueError("sims.verilatorThreads must be positive")
    meta = {
        "index": 0,
        "thread_id": 0,
        "role": pkg,
        "pkg": pkg,
        "config_path": _rel(bbdir, raw["_file"]),
        "balldomain_base_dir": _derive._balldomain_base_dir(bbdir, raw, pkg),
    }
    _fill_thread(b.threads.add(), raw, meta, bbdir, hardware=True)
    for ball in _derive._bemu_balls(bbdir, [{"config": raw}]):
        e = b.bemu.balls.add()
        e.emu_class = ball["emu_class"]
        e.component = ball["component"]
        e.emu_lib = ball["emu_lib"]

    # The entry of a thread alone: its RTL, model and workload top. A prefab
    # thread runs on the prefab entry, whose classpath holds every prefab
    # thread; a design's thread on its own, <thread>/sims/thread, on the
    # design's classpath. Its output is under the thread's own directory, and
    # so are its tests and their regression lists (<thread>/workloads/thread).
    entry = (
        Path("examples/prefab/sims/thread")
        if root.is_relative_to(bbdir / "examples/prefab/threads")
        else Path(_rel(bbdir, root)) / "sims/thread"
    )
    target = b.simulation
    target.selection = config["sims"]["selection"]
    target.output = str(Path(_rel(bbdir, root)) / "output/sims" / target.selection)
    target.directory = str(entry)
    target.arch.directory = str(entry / "arch")
    target.arch.module = "simulation"
    target.arch.config = str(Path(target.output) / "generated/design.pb")
    target.entry_class = "framework.thread.sims.Elaborate"
    target.top = "ThreadAlone"
    target.emu = str(entry / "emu")
    target.workloads = str(entry / "workloads")
    target.regression = str(Path(_rel(bbdir, root)) / "workloads/thread/regression")
    manifest = bbdir / target.emu / "Cargo.toml"
    target.emu_package = tomllib.loads(manifest.read_text())["package"]["name"]
    for path in (
        Path(target.arch.directory) / "build.sc",
        Path(target.workloads) / "CMakeLists.txt",
        Path(_rel(bbdir, root)) / "workloads/thread/CMakeLists.txt",
    ):
        if not (bbdir / path).is_file():
            raise FileNotFoundError(bbdir / path)
    return b


def write_thread_alone_pb(config_path: Path, out_path: Path, bbdir: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    design = fill_thread_design(config, bbdir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists() or pb.Design.FromString(out_path.read_bytes()) != design:
        out_path.write_bytes(design.SerializeToString(deterministic=True))
    return out_path


def write_ball_pb(config_path: Path, out_path: Path, bbdir: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    design = fill_ball_design(config, bbdir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists() or pb.Design.FromString(out_path.read_bytes()) != design:
        out_path.write_bytes(design.SerializeToString(deterministic=True))
    return out_path


def write_design_pb(
    config_path: Path,
    derived_path: Path,
    out_path: Path,
    bbdir: Path,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    derived = json.loads(derived_path.read_text(encoding="utf-8"))
    design = fill_design(config, derived, bbdir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists() or pb.Design.FromString(out_path.read_bytes()) != design:
        out_path.write_bytes(design.SerializeToString(deterministic=True))
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a configured hart protobuf using the standard TOML expansion"
    )
    parser.add_argument("--hart", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    options = parser.parse_args()
    source = Path(__file__).with_name("1_toml2json.py")
    spec = importlib.util.spec_from_file_location("toml_expansion", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = write_hart_pb(module.toml2json(options.hart), options.output, options.repo)
    print(output)
