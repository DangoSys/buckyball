import importlib.util
import json
import os
import shlex
import sys
from pathlib import Path

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

scripts_dir = os.path.join(os.path.dirname(__file__), "scripts")

from utils.event_common import check_result, get_origin_trace_id
from utils.path import get_buckyball_path
from utils.stream_run import stream_run_logger_async

config = {
    "name": "config-install",
    "description": "install all chip configs under examples/chips",
    "flows": ["config"],
    "triggers": [queue("config.install")],
    "enqueues": [],
}


def _load_script(name: str):
    path = os.path.join(scripts_dir, name)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin_tid = get_origin_trace_id(input_data, ctx)
    bbdir = Path(get_buckyball_path())
    chips = bbdir / "examples" / "chips"

    tops = sorted(chips.glob("*/configs/chip.toml"))
    if not tops:
        raise ValueError(f"no configs/chip.toml under {chips}")

    toml2json = _load_script("1_toml2json.py")
    derive = _load_script("2_parameter_derivation.py")

    # step 1: generate chip config JSONs from toml files
    json_outs: list[Path] = []
    for top in tops:
        data = toml2json.toml2json(top)
        out = top.parent / "generated" / "config" / "config.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        json_outs.append(out)
        ctx.logger.info(f"Wrote chip config JSON: {out}")

    # step 2: derive computed parameters from config JSONs
    derived_outs: list[Path] = []
    for top, json_path in zip(tops, json_outs):
        chip = top.parent.parent.name
        data = json.loads(json_path.read_text(encoding="utf-8"))
        derived_path = json_path.parent / "derived.json"
        derive.write_derived(data, bbdir, chip, derived_path)
        derived_outs.append(derived_path)
        ctx.logger.info(f"Wrote derived config: {derived_path}")

    # step 3.1: generate chip_pb2.py from chip.proto
    scripts = Path(scripts_dir)
    cmd = (
        f"protoc -I{shlex.quote(str(scripts / 'proto'))} "
        f"--python_out={shlex.quote(str(scripts))} "
        f"{shlex.quote(str(scripts / 'proto' / 'chip.proto'))}"
    )
    ctx.logger.info(f"Running protoc: {cmd}")
    result = await stream_run_logger_async(
        cmd=cmd,
        logger=ctx.logger,
        stdout_prefix="protoc",
        stderr_prefix="protoc",
        task_scope=origin_tid,
    )
    if result.returncode != 0:
        raise RuntimeError(f"protoc failed with returncode={result.returncode}")
    ctx.logger.info(f"Generated chip python bindings: {scripts / 'chip_pb2.py'}")

    # step 3.2: generate chip.pb for all components build
    json2proto = _load_script("3_json2proto.py")
    pb_outs: list[Path] = []
    for json_path, derived_path in zip(json_outs, derived_outs):
        out = json_path.parent / "chip.pb"
        json2proto.write_chip_pb(json_path, derived_path, out, bbdir)
        pb_outs.append(out)
        ctx.logger.info(f"Wrote chip protobuf: {out}")

    # step 4: translate PB to different build systems
    step4 = _load_script("4_install.py")
    installed: list[dict[str, str]] = []
    for top, src_pb, json_path in zip(tops, pb_outs, json_outs):
        chip_name = top.parent.parent.name
        chip = step4.load_chip(src_pb)
        gen = top.parent / "generated"

        # A. TO Arch BUILD SYSTEM
        arch = step4.install_arch(src_pb, gen)
        ctx.logger.info(f"Installed arch pb: {arch}")

        # B. TO BEMU BUILD SYSTEM
        bemu = step4.install_bemu(chip, bbdir, gen)
        ctx.logger.info(f"Installed bemu crate: {bemu}")

        # C. TO Workload BUILD SYSTEM
        workload = step4.install_workload(chip, bbdir, chip_name, gen)
        ctx.logger.info(f"Installed workload defs: {workload}")

        default_design = Path(json.loads(json_path.read_text(encoding="utf-8"))["designs"]["_file"]).resolve()
        designs_dir = top.parent / "designs"
        if not designs_dir.is_dir():
            raise ValueError(f"missing {designs_dir}")
        for design in sorted(designs_dir.glob("*.toml")):
            if design.resolve() == default_design:
                continue
            vdata = toml2json.toml2json(top, design=design)
            vjson = gen / "config" / f"{design.stem}.json"
            vjson.write_text(json.dumps(vdata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            vderived = gen / "config" / f"{design.stem}.derived.json"
            derive.write_derived(vdata, bbdir, chip_name, vderived)
            vpb = json2proto.write_chip_pb(vjson, vderived, gen / f"{design.stem}.pb", bbdir)
            ctx.logger.info(f"Installed arch variant pb: {vpb}")

        installed.append(
            {
                "chip": chip_name,
                "arch": str(arch),
                "bemu": str(bemu),
                "workload": str(workload),
            }
        )

    await check_result(
        ctx,
        0,
        continue_run=False,
        extra_fields={
            "task": "install",
            "json": [str(p) for p in json_outs],
            "derived": [str(p) for p in derived_outs],
            "pb2": str(scripts / "chip_pb2.py"),
            "install": installed,
        },
        trace_id=origin_tid,
    )
