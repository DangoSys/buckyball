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
    msg.mem.addr_len = d["mem"]["addrLen"]


def _fill_ball(msg: pb.BallDomain, d: dict[str, Any], bbdir: Path) -> None:
    for ball in d["balls"]:
        instance = msg.balls.add()
        instance.ball_id = ball["id"]
        instance.ball_name = ball["name"]
        instance.ball_dir = ball["component"]
        classes = []
        root = bbdir / "examples/components/balls" / ball["component"] / "arch"
        for source in sorted(root.rglob("*Ball.scala")):
            text = source.read_text()
            names = re.findall(
                r"class (\w+)\([^{}]*?\)\s+extends Module\s+with HasBlink\b", text, re.S
            )
            if names:
                package = re.search(r"(?m)^package ([\w.]+)", text).group(1)
                classes.extend(f"{package}.{name}" for name in names)
        if len(classes) != 1:
            raise ValueError(
                f"{ball['component']}: expected one Ball implementation, found {classes}"
            )
        instance.rtl_class = classes[0]
        port = ball["port"]
        instance.in_bw = port["inBW"]
        instance.out_bw = port["outBW"]
        instance.mmio_read_bw = port["mmioReadBW"]
        instance.mmio_write_bw = port["mmioWriteBW"]
        pack_parameters(
            instance,
            {key: value for key, value in ball["param"].items() if key != "_file"},
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
    msg.vbank_id_upper_bound = d["vbankIdUpperBound"]
    msg.shared_bank_id_base = d["sharedBankIdBase"]
    msg.iter_len = d["iterLen"]
    msg.sub_rob_enable = d["subRobEnable"]
    msg.sub_rob_depth = d["subRobDepth"]
    msg.bank_alias_enable = d["bankAliasEnable"]


def _fill_gpdomain(msg: pb.GpDomainConfig, d: dict[str, Any], bbdir: Path) -> None:
    if type(d["enable"]) is not bool:
        raise ValueError("gpdomain.enable must be an explicit boolean")
    msg.enable = d["enable"]
    for key in ("laneNumber", "vLen", "eLen", "iBufWords", "memoryPorts"):
        if type(d[key]) is not int or d[key] <= 0:
            raise ValueError(f"gpdomain.{key} must be a positive integer")
    msg.source_path = _rel(bbdir, d["_file"])
    msg.lane_number = d["laneNumber"]
    msg.v_len = d["vLen"]
    msg.e_len = d["eLen"]
    msg.i_buf_words = d["iBufWords"]
    msg.memory_ports = d["memoryPorts"]


def _fill_core_params(msg: pb.CoreParamConfig, d: dict[str, Any]) -> None:
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
    source = (Path(raw["_file"]).parent / raw["source"]).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"missing component source directory: {source}")
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
    b.design_path = derived["design_path"]
    b.n_cores = derived["n_cores"]
    b.includes.extend(derived["includes"])

    for key in ("ddr", "clint", "plic", "scu"):
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
    image[8:16] = b.n_cores.to_bytes(8, "little")
    image[16:24] = bootrom["entry"].to_bytes(8, "little")
    b.device.bootrom.base = bootrom["base"]
    b.device.bootrom.bytes = bootrom["bytes"]
    b.device.bootrom.image.extend(byte if byte < 128 else byte - 256 for byte in image)
    b.includes.append(_rel(bbdir, image_path))

    ParseDict(
        {key: _hardware_fields(config["sims"][key]) for key in ("diffTest", "memory")},
        b.sims,
    )
    if not b.device.IsInitialized() or not b.sims.IsInitialized():
        raise ValueError("Device and simulation parameters must be complete")

    root = Path(derived["design_path"]).parent.parent
    for name, registration in config["sims"]["targets"].items():
        if name == "_file":
            continue
        if name != "device" and not re.fullmatch(r"core/[A-Za-z0-9_-]+", name):
            raise ValueError(f"invalid simulation target: {name}")
        if name.startswith("core/") and "regression" in registration:
            raise ValueError(f"{name}: regression belongs to the device entry")
        target = b.simulations[name]
        target.directory = str(root / "sims" / name)
        for field in ("emu", "stack", "workloads"):
            setattr(target, field, str(root / "sims" / name / field))
        target.arch.directory = str(root / "sims" / name / "arch")
        target.arch.module = "simulation"
        target.arch.config = str(root / "output/generated/design.pb")
        target.entry_class = registration["config"]
        target.top = "BBSimHarness"
        target.stack_name = registration["stack_name"]
        manifest = bbdir / target.emu / "Cargo.toml"
        target.emu_package = tomllib.loads(manifest.read_text())["package"]["name"]
        for resource in ("regression", "kernel", "tapeout"):
            if resource in registration:
                target.resources[resource] = str(root / registration[resource])
        for entry in (
            Path(target.arch.directory) / "build.sc",
            Path(target.stack) / "CMakeLists.txt",
            Path(target.workloads) / "CMakeLists.txt",
        ):
            if not (bbdir / entry).is_file():
                raise FileNotFoundError(bbdir / entry)
    from verification import ip_registrations

    profiles = ip_registrations(
        bbdir, config["sims"]["verification"]["registry"]["ips"]
    )
    b.verification.ips.extend(dict.fromkeys(profile.ip for profile in profiles))
    b.verification.profiles.extend(profiles)

    for raw in _derive.configured_cores(config["device"]):
        _append_core(b, raw, bbdir)

    if "sharedBankFabric" in config["device"]:
        fabric = config["device"]["sharedBankFabric"]
        harts = {core.hart_id for core in b.device.cores}
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
        core.accelerator.balldomain
        for core in b.device.cores
        if core.HasField("accelerator")
    ]
    domains.extend(thread.balldomain for thread in b.threads)
    b.verification.balls.extend(
        dict.fromkeys(ball.ball_dir for domain in domains for ball in domain.balls)
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
            if compiled[name].config.signature != instance.signature:
                raise ValueError(
                    f"compiler {name}: instances have different parameters; declare distinct compiler names"
                )
            continue
        p = b.profiles.add()
        compiled[name] = p
        p.name = name
        p.pkg = profile["component"]
        p.compiler = "compiler"
        bank = targets[name]["bank"]
        p.bank_num, p.bank_width, p.bank_entries = (
            bank["num"],
            bank["width"],
            bank["entries"],
        )
        p.ball_ctest_dirs.extend(targets[name]["ball_ctest_dirs"])
        p.config.CopyFrom(instance)

    for name, selected in config["workloads"].get("profiles", {}).items():
        if name == "_file":
            continue
        if (
            type(selected) is not int
            or selected < 0
            or not any(
                c.hart_id == selected and c.profile == name for c in b.device.cores
            )
        ):
            raise ValueError(
                f"compiler {name}: test hart is not a direct accelerator binding"
            )
        compiled[name].test_core = selected

    bemu = derived["bemu"]
    for ball in bemu["balls"]:
        e = b.bemu.balls.add()
        e.ball_class = ball["ball_class"]
        e.ball_dir = ball["ball_dir"]
        e.emu_lib = ball["emu_lib"]

    harts = {core.hart_id for core in b.device.cores}
    tests = config["workloads"]
    for key, field in (("controlHart", "control_hart"), ("reportHart", "report_hart")):
        hart = tests[key]
        if type(hart) is not int or hart not in harts:
            raise ValueError(f"workload {key}: unknown hart {hart!r}")
        setattr(b.workload, field, hart)

    for key, val in derived["workload"]["cmake_param"].items():
        b.workload.cmake_param[key] = val
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


def _append_core(design: pb.Design, raw: dict[str, Any], bbdir: Path) -> None:
    core = design.device.cores.add()
    core.path = _rel(bbdir, raw["_file"])
    component = raw.get("component")
    if not isinstance(component, str) or not component:
        raise ValueError("Core component must be explicitly named")
    core.component = component
    core.profile = raw["compiler"]["name"] if "compiler" in raw else ""
    core.hart_id = raw["hartId"]
    core.reset_vector = raw["resetVector"]
    core.mtime_address = raw["mtimeAddress"]
    for key in ("memory", "fetch", "fpu", "interconnect", "c2c", "controller"):
        ParseDict(_hardware_fields(raw[key]), getattr(core, key))
    if "tbus" in raw:
        ParseDict(_hardware_fields(raw["tbus"]), core.tbus)
    if "dmaCache" in raw:
        ParseDict(_hardware_fields(raw["dmaCache"]), core.dma_cache)
    if not core.IsInitialized():
        raise ValueError("Core IP configuration is incomplete")
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
        core.implementation.Pack(implementation, deterministic=True)
    if "param" in raw or "implementation" not in raw:
        _fill_core_params(core.param, raw["param"])
    if "tss" in raw:
        _fill_spm(core.tss, raw["tss"])
    shared = raw["sharedMem"]
    core.virtual_bank_count = shared["virtualBankCount"]
    core.shared_mem.enable = shared["enable"]
    core.shared_mem.entries = shared["entries"]
    core.shared_mem.bank_entries = shared["bankEntries"]
    core.shared_mem.bank_width = shared["bankWidth"]
    core.shared_mem.input_channels = shared["inputChannels"]
    core.shared_mem.default_group_count = shared["defaultGroupCount"]
    core.shared_mem.virtual_bank_count = shared["virtualBankCount"]
    core.shared_mem.mem_cycles = shared["memCycles"]
    if "accelerator" in raw:
        acc = raw["accelerator"]
        _fill_ball(core.accelerator.balldomain, acc["balldomain"], bbdir)
        _fill_mem(core.accelerator.mem, acc["memdomain"], bbdir)
        _fill_frontend(core.accelerator.frontend, acc["frontend"], bbdir)
        _fill_gpdomain(core.accelerator.gpdomain, acc["gpdomain"], bbdir)
        ParseDict(_hardware_fields(acc["transport"]), core.accelerator.transport)
    threads = raw["threads"]
    ids = [thread["id"] for thread in threads]
    if any(type(value) is not int or value < 0 for value in ids) or len(
        set(ids)
    ) != len(ids):
        raise ValueError(
            "Core-local thread IDs must be explicit, non-negative and unique"
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
        core.thread_indices.append(index)
    # The signature names the kind of core software runs on; where an instance
    # sits (hart, config file, NoC nodes) is not part of it.
    hardware = pb.CorePlacement()
    hardware.CopyFrom(core)
    for field in ("hart_id", "path", "thread_indices"):
        hardware.ClearField(field)

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
        for index in core.thread_indices
    )
    core.signature = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def write_core_pb(raw: dict[str, Any], out_path: Path, bbdir: Path) -> Path:
    design = pb.Design(name=f"hart-{raw['hartId']}", n_cores=1)
    _append_core(design, raw, bbdir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    out_path.write_bytes(design.SerializeToString())
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a configured core protobuf using the standard TOML expansion"
    )
    source_kind = parser.add_mutually_exclusive_group(required=True)
    source_kind.add_argument("--core", type=Path)
    source_kind.add_argument("--design", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    options = parser.parse_args()
    source = Path(__file__).with_name("1_toml2json.py")
    spec = importlib.util.spec_from_file_location("toml_expansion", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if options.core is not None:
        output = write_core_pb(
            module.toml2json(options.core), options.output, options.repo
        )
    else:
        data = module.design2json(options.design)
        derived = _derive.derive(
            data, options.repo, options.design.resolve().parent.parent.name
        )
        design = fill_design(data, derived, options.repo)
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_bytes(design.SerializeToString(deterministic=True))
        output = options.output
    print(output)
