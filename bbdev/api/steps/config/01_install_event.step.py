import importlib.util
import json
import os
import shlex
import sys
import tempfile
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
    "description": "install all design configs under examples/designs",
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
    designs = bbdir / "examples" / "designs"

    selected = input_data.get("design")
    tops = (
        [designs / selected / "configs" / "design.toml"]
        if selected
        else sorted(designs.glob("*/configs/design.toml"))
    )
    if any(not top.is_file() for top in tops):
        raise ValueError(f"design {selected!r} has no configs/design.toml assembly")
    if not tops:
        raise ValueError(f"no configs/design.toml under {designs}")

    toml2json = _load_script("1_toml2json.py")
    derive = _load_script("2_parameter_derivation.py")

    # step 1: generate design config JSONs from toml files
    json_outs: list[Path] = []
    for top in tops:
        data = toml2json.design2json(top)
        out = top.parent.parent / "output" / "generated" / "config" / "config.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        json_outs.append(out)
        ctx.logger.info(f"Wrote design config JSON: {out}")

    # step 2: derive computed parameters from config JSONs
    derived_outs: list[Path] = []
    for top, json_path in zip(tops, json_outs):
        design = top.parent.parent.name
        data = json.loads(json_path.read_text(encoding="utf-8"))
        derived_path = json_path.parent / "derived.json"
        derive.write_derived(data, bbdir, design, derived_path)
        derived_outs.append(derived_path)
        ctx.logger.info(f"Wrote derived config: {derived_path}")

    # step 3.1: generate design_pb2.py from design.proto
    scripts = Path(scripts_dir)
    with tempfile.TemporaryDirectory(prefix="bbdev-proto-") as generated:
        schemas = [
            scripts / "proto" / "design.proto",
            scripts / "proto" / "hardware.proto",
            *sorted(
                Path(bbdir).glob("examples/components/balls/*/configs/*_params.proto")
            ),
            *sorted(
                Path(bbdir).glob("examples/components/cores/*/configs/*_params.proto")
            ),
        ]
        includes = sorted({schema.parent for schema in schemas})
        cmd = " ".join(
            [
                "protoc",
                *(f"-I{shlex.quote(str(directory))}" for directory in includes),
                f"--python_out={shlex.quote(generated)}",
                *(shlex.quote(str(schema)) for schema in schemas),
            ]
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
        for source in Path(generated).glob("*_pb2.py"):
            content = source.read_bytes()
            destination = scripts / source.name
            if not destination.is_file() or destination.read_bytes() != content:
                destination.write_bytes(content)
    ctx.logger.info(f"Generated design python bindings: {scripts / 'design_pb2.py'}")

    # step 3.2: design.pb, the one input of every build system
    json2proto = _load_script("3_json2proto.py")
    pb_outs: list[Path] = []
    for json_path, derived_path in zip(json_outs, derived_outs):
        out = json_path.parent.parent / "design.pb"
        json2proto.write_design_pb(json_path, derived_path, out, bbdir)
        pb_outs.append(out)
        ctx.logger.info(f"Wrote design protobuf: {out}")

    await check_result(
        ctx,
        0,
        continue_run=False,
        extra_fields={
            "task": "install",
            "json": [str(p) for p in json_outs],
            "derived": [str(p) for p in derived_outs],
            "pb2": str(scripts / "design_pb2.py"),
            "design_pb": [str(p) for p in pb_outs],
        },
        trace_id=origin_tid,
    )
