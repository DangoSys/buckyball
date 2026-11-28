import asyncio
import fcntl
import json
import os
import re
import subprocess
import sys
import tomllib
import uuid
from datetime import datetime, timezone
from pathlib import Path

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from steps.bebop.p2e.scripts.runtime_case import case_ready, trace_modes
from utils.event_common import check_result, get_origin_trace_id, require_target
from utils.run_step import run as run_step
from utils.workload_manifest import resolve_workload_toml
from utils.verilator_turn import verilator_turn
from utils.path import design_output_root, get_buckyball_path

config = {
    "name": "bench-tool",
    "description": "measure a tool's speed on a device into a bbbench snapshot",
    "flows": ["bench"],
    "triggers": [queue("bench.tool")],
    "enqueues": [
        "bebop.bemu.batch",
        "bebop.bemu.sim",
        "bebop.verilator.clean",
        "bebop.verilator.verilog",
        "bebop.verilator.build",
        "bebop.verilator.batch",
        "bebop.p2e.runworkload",
    ],
}


async def bemu(
    ctx, save: Path, device: str, design: str, data: dict, publish
) -> tuple[list[dict], dict]:
    """Measure each registered end-to-end workload on this device, in isolation."""
    target = {"device": device}
    target["target"] = require_target(target, ("device",))
    (spec,) = (
        d
        for d in json.loads((save / "bbmap.json").read_text())["designs"]
        if d["design"] == design
    )
    manifest = Path(
        resolve_workload_toml(design, "bemu", "linux-tests", selection=target["target"])
    )
    cases = [
        (name, {"test": name})
        for name in tomllib.loads(manifest.read_text())["workloads"]["tests"]
    ]
    for kind, names in (("model", spec["models"]), ("bench", spec["benches"])):
        for name in names:
            for os_name in ("baremetal", "linux"):
                cases.append((f"{name}-{os_name}", {kind: name, "os": os_name}))
    trace = (
        {flag: True for flag in ("itrace", "mtrace", "ctrace", "banktrace")}
        if data["all-trace"]
        else {}
    )
    workloads = []
    progress = {
        "target_hz": 200_000_000,
        "min_elapsed_s": 1,
        "total": len(cases),
        "status": "running",
    }
    publish(workloads, progress)
    for name, product in cases:
        state, body = await run_step(
            ctx, "bebop.bemu.sim", {**target, **product, **trace}, 1800
        )
        row = {"workload": name, "status": state}
        if "log_dir" in body:
            log = Path(body["log_dir"])
            row["log"] = log.name
            output = (log / "run.log").read_text(errors="replace")
            wall = re.search(r"BEMU wall time: ([\d.]+)s", output)
            if wall:
                row["elapsed_s"] = float(wall[1])
            if state == "success":
                states = [f.read_text() for f in log.glob("id-*/cpu-state.log")]
                counts = [re.search(r"elapsed_cycles=(\d+)", line) for line in states]
                cycles = max((int(m[1]) for m in counts if m), default=0)
                if cycles > 0 and all(counts) and row.get("elapsed_s", 0) > 0:
                    row["cycles"] = cycles
                    row["frequency_hz"] = row["cycles"] / row["elapsed_s"]
                    if row["elapsed_s"] < progress["min_elapsed_s"]:
                        row["status"] = "too_short"
                    else:
                        row["status"] = (
                            "pass"
                            if row["frequency_hz"] > progress["target_hz"]
                            else "below_target"
                        )
            else:
                error = re.search(
                    r"^(?:FAIL(?:ED_TESTS)?\b|Error:|.*panicked at).+", output, re.M
                )
                if error:
                    row["error"] = error[0].strip()
        if row["status"] == "success":
            row.update(status="failure", error="Incomplete end-to-end timing")
        elif state != "success" and "error" not in row:
            row["error"] = body.get("error") or f"{body.get('task', 'BEMU')}: {state}"
        workloads.append(row)
        publish(workloads, progress)
        ctx.logger.info(f"BEMU workload result {device} {json.dumps(row)}")
    return workloads, {
        **progress,
        "status": (
            "pass"
            if any(r["status"] == "pass" for r in workloads)
            and all(r["status"] in ("pass", "too_short") for r in workloads)
            else "failure"
        ),
    }


async def run(ctx, topic: str, data: dict) -> dict:
    """Run one step as its own task and wait for its end."""
    trace = str(uuid.uuid4())
    await ctx.enqueue({"topic": topic, "data": {**data, "_trace_id": trace}})
    while True:
        for state in ("success", "failure", "cancelled"):
            try:
                value = await ctx.state.get(trace, state)
            except TimeoutError:
                # The state service of a loaded host answers late; the next
                # poll asks again.
                break
            if value:
                body = value.get("body", value)
                if state != "success" and "log_dir" not in body:
                    raise RuntimeError(f"{topic}: {body.get('error', state)}")
                return body
        await asyncio.sleep(10)


async def verilator(
    ctx, save: Path, device: str, design: str, data: dict
) -> tuple[list[dict], dict]:
    """Verilator's speed is measured, not read from a regression: the model
    of the current tree built (with DiffTest under --diff), and the first ten
    workloads of the device's bare list run on it (with every trace under
    --all-trace). A frequency needs no more, and the whole regression of a
    large device takes hours."""
    diff = data["diff"]
    target = {"device": device}
    target["target"] = require_target(target, ("device",))
    # Models take many cores and their host threads spin while they wait:
    # measurements and regressions take turns, first come, first served
    # (utils/verilator_turn.py).
    turn = verilator_turn(Path(get_buckyball_path()))
    await asyncio.to_thread(turn.__enter__)
    try:
        await run(ctx, "bebop.verilator.clean", target)
        await run(ctx, "bebop.verilator.verilog", {**target, "diff": diff})
        await run(ctx, "bebop.verilator.build", {**target, "diff": diff, "jobs": 100})
        body = await run(
            ctx,
            "bebop.verilator.batch",
            {
                **target,
                "test": "bare-tests",
                "diff": diff,
                "all-trace": data["all-trace"],
                "jobs": 16,
                "limit": 10,
            },
        )
    finally:
        turn.__exit__(None, None, None)
    log = Path(body["log_dir"])
    return passed(log), {"log": log.name}


async def p2e(
    ctx, save: Path, device: str, design: str, data: dict
) -> tuple[list[dict], dict]:
    """P2E's speed on a device: its bare-metal whole models. Fastest, nothing
    is run here: the timeline's fpga-model step records each run's cycles
    and wall time, read from the newest snapshot in which it ended. With
    every trace (--all-trace) or DiffTest (--diff), those models run again
    on the device's newest completed bitstream built so (bebop-p2e
    --buildbitstream --itrace --mtrace, or --diff), every trace on or
    DiffTest checking."""
    node, ran = None, None
    for entry in json.loads((save / "index.json").read_text())["snapshots"]:
        if not entry["id"].startswith(f"designs/{design}/"):
            continue
        for n in json.loads((save / entry["id"] / "design.json").read_text())["nodes"]:
            if n["id"] == "fpga-model" and n["status"] in ("pass", "failure"):
                node, ran = n, entry["id"]
    if node is None:
        raise RuntimeError(f"{device}: no fpga-model step ended in its timeline")
    models = [
        m for m in node["models"] if m["status"] == "pass" and m["os"] == "baremetal"
    ]
    if not data["all-trace"] and not data["diff"]:
        return [
            {
                "workload": f"{m['model']}-baremetal",
                "elapsed_s": m["elapsed_s"],
                "cycles": m["cycles"],
            }
            for m in models
        ], {"timeline": ran}
    target = {"device": device}
    target["target"] = require_target(target, ("device",))
    repo = Path(get_buckyball_path())
    output = Path(design_output_root(repo, design, selection=target["target"]))
    built = {"btrace", "btrace_nb_v1"} if data["diff"] else {"itrace", "mtrace"}
    cases = sorted(
        bit
        for bit in (output / "p2e").glob("*/fpgaCompDir/bitstream.bit")
        if case_ready(bit)
        and (
            trace_modes(str(bit)) & built
            if data["diff"]
            else built <= trace_modes(str(bit))
        )
    )
    if not cases:
        raise RuntimeError(
            f"{device}: no completed bitstream built with "
            f"{'--diff' if data['diff'] else '--itrace --mtrace'}"
        )
    bit = cases[-1]
    flags = (
        {"diff": True}
        if data["diff"]
        else {"itrace": True, "mtrace": True, "pmctrace": True, "ctrace": True}
    )
    workloads, logs = [], []
    # The board is exclusive; the runtime also writes flags and Tcl into its
    # case (the locks of the fpga-model step).
    with (repo / "log/fpga-0.A.lock").open("a") as board, (
        bit.parent.parent / "bench.lock"
    ).open("a") as case:
        await asyncio.to_thread(fcntl.flock, board, fcntl.LOCK_EX)
        await asyncio.to_thread(fcntl.flock, case, fcntl.LOCK_EX)
        for m in models:
            body = await run(
                ctx,
                "bebop.p2e.runworkload",
                {
                    **target,
                    "bitstream": str(bit),
                    "fpga-location": "0.A",
                    "model": m["model"],
                    "os": "baremetal",
                    **flags,
                },
            )
            log = Path(body["log_dir"])
            logs.append(log.name)
            uart = "\n".join(p.read_text() for p in sorted(log.glob("uart_hart_*.log")))
            cycles = re.findall(r"Inference cycles: (\d+)", uart)
            if not body.get("success") or not cycles:
                continue
            (seconds,) = re.findall(
                r"Total P2E command: ([\d.]+) s", (log / "p2e_timing.log").read_text()
            )
            workloads.append(
                {
                    "workload": f"{m['model']}-baremetal",
                    "elapsed_s": float(seconds),
                    "cycles": int(cycles[0]),
                }
            )
    return workloads, {
        "timeline": ran,
        "bitstream": str(bit.relative_to(repo)),
        "logs": logs,
    }


def passed(log: Path) -> list[dict]:
    """The workloads of a batch run that passed: their wall time and
    simulated cycles, BEMU's the slowest hart's latest elapsed_cycles in its
    cpu-state.log, Verilator's the clock count the harness prints at its
    end."""
    workloads = []
    for summary in sorted(log.glob("*/summary.json")):
        s = json.loads(summary.read_text())
        if s["status"] != "pass":
            continue
        states = sorted(summary.parent.glob("log/id-*/cpu-state.log"))
        simulated = (
            max(
                int(re.search(r"elapsed_cycles=(\d+)", state.read_text()).group(1))
                for state in states
            )
            if states
            else int(
                re.search(
                    r"^Simulation cycles: (\d+)$",
                    (summary.parent / "stdout.log").read_text(),
                    re.M,
                ).group(1)
            )
        )
        workloads.append(
            {
                "workload": s["workload_name"],
                "status": "pass",
                "elapsed_s": float(s["elapsed_sec"].removesuffix("s")),
                "cycles": simulated,
            }
        )
    return workloads


# The tools measured, by the name --tool takes: the name the pages show and
# how the simulations its frequency is taken over are found.
tools = {
    "bemu": ("BEMU", bemu),
    "verilator": ("Verilator", verilator),
    "p2e": ("FPGA", p2e),
}


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def register(save: Path, entry: dict) -> None:
    """Append the snapshot to index.json, under the lock every bench takes."""
    with open(save / "index.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index_path = save / "index.json"
        index = json.loads(index_path.read_text())
        index["snapshots"].append(entry)
        temporary = index_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
        temporary.replace(index_path)


async def measure(ctx, data: dict, device: str) -> dict:
    """A tool's speed on a device (simulated cycles per second of wall time)
    over its simulations that passed, as a snapshot. Correctness is the
    timeline's matter: a simulation that did not pass is left out, the rest
    are summed."""
    repo = Path(get_buckyball_path())
    tool = data["tool"]
    name, simulations = tools[tool]
    # The flag the tool ran with, if any: its snapshots apart from the
    # fastest ones, tools/<tool>-<flag>/.
    (flag,) = [f for f in ("all-trace", "diff") if data[f]] or [None]
    (owner,) = [
        p.parent.parent for p in (repo / "examples/designs").glob(f"*/devices/{device}")
    ]
    design = owner.name
    save = repo / "thirdparty/bbbench/save"
    commit = git(repo, "rev-parse", "HEAD")
    dirty = bool(git(repo, "status", "--porcelain", "--untracked-files=no"))
    taken = datetime.now(timezone.utc)
    # save/<kind>/<subject>/<time>-<commit>[-dirty]; the index id is that path.
    snapshot = (
        f"tools/{tool}{'-' + flag if flag else ''}/{design}/{taken:%Y-%m-%d-%H%M%S}-{commit[:7]}"
        + ("-dirty" if dirty else "")
    )
    directory = save / snapshot
    directory.mkdir(parents=True, exist_ok=False)
    report = directory / "tools.json"

    def publish(workloads, source):
        first = not report.exists()
        measured = (
            [w for w in workloads if w["status"] in ("pass", "below_target")]
            if tool == "bemu"
            else workloads
        )
        value = {
            "commit": commit,
            "dirty": dirty,
            "taken": taken.isoformat(),
            "design": design,
            "tool": name,
            **({"flag": flag} if flag else {}),
            **source,
            "cycles": sum(w["cycles"] for w in measured if "cycles" in w),
            "elapsed_s": sum(
                w["elapsed_s"] for w in measured if "cycles" in w and "elapsed_s" in w
            ),
            "workloads": workloads,
        }
        temp = report.with_suffix(".tmp")
        temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temp.replace(report)
        if first:
            register(
                save,
                {
                    "id": snapshot,
                    "commit": commit,
                    "dirty": dirty,
                    "taken": taken.isoformat(),
                    "files": ["tools.json"],
                },
            )

    if tool == "bemu":
        try:
            workloads, source = await bemu(ctx, save, device, design, data, publish)
        except Exception as error:
            if report.exists():
                partial = json.loads(report.read_text())
                publish(
                    partial.pop("workloads"),
                    {**partial, "status": "failure", "error": str(error)},
                )
            raise
    else:
        workloads, source = await simulations(ctx, save, device, design, data)
    if not workloads:
        raise RuntimeError(f"{device}: no {name} workloads")
    publish(workloads, source)
    return {"snapshot": snapshot, "status": source.get("status", "pass")}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin = get_origin_trace_id(input_data, ctx)
    # all: every device of every design, a snapshot each.
    devices = (
        sorted(
            p.name
            for p in Path(get_buckyball_path()).glob("examples/designs/*/devices/*")
        )
        if input_data["device"] == "all"
        else [input_data["device"]]
    )
    snapshots, errors = {}, {}
    for device in devices:
        try:
            result = await measure(ctx, input_data, device)
            snapshots[device] = result["snapshot"]
            if result["status"] != "pass":
                errors[device] = "End-to-end BEMU speed target not met"
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            errors[device] = str(error)
    await check_result(
        ctx,
        1 if errors else 0,
        continue_run=False,
        extra_fields={"task": "tools", "snapshots": snapshots, "errors": errors},
        trace_id=origin,
    )
