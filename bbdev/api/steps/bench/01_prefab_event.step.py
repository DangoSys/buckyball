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

from utils.event_common import check_result, get_origin_trace_id, require_target
from utils.path import get_buckyball_path, simulation_target

config = {
    "name": "bench-prefab",
    "description": "evaluate prefab components into a bbbench snapshot",
    "flows": ["bench"],
    "triggers": [queue("bench.prefab")],
    "enqueues": [
        "dc.verilog",
        "uvm.run",
        "workload.build",
        "bebop.bemu.batch",
        "bebop.verilator.verilog",
        "bebop.verilator.build",
        "bebop.verilator.batch",
    ],
}

# Runs at once: a Ball's DC or UVM run, a thread's or a hart's whole
# evaluation; DC uses its tapeout contract's max_threads.
parallel = 4


def all_balls(repo: Path) -> list[str]:
    """Every Ball: the prefab ones by name, a design's as <design>/balls/<ball>."""
    prefab = sorted(
        p.name
        for p in (repo / "examples/prefab/balls").iterdir()
        if (p / "configs/default.toml").is_file()
    )
    designs = sorted(
        str(p.relative_to(repo / "examples/designs"))
        for p in (repo / "examples/designs").glob("*/balls/*")
        if (p / "configs/default.toml").is_file()
    )
    return prefab + designs


def all_threads(repo: Path) -> list[str]:
    """Every thread that runs alone: the prefab ones by name, a design's as
    <design>/threads/<thread>."""
    prefab = sorted(
        p.name
        for p in (repo / "examples/prefab/threads").iterdir()
        if (p / "configs/default.toml").is_file()
    )
    designs = sorted(
        str(p.relative_to(repo / "examples/designs"))
        for p in (repo / "examples/designs").glob("*/threads/*")
        if (p / "configs/default.toml").is_file()
    )
    return prefab + designs


def all_harts(repo: Path) -> list[str]:
    """Every hart that runs alone: the prefab ones by name, a design's as
    <design>/harts/<hart>."""
    prefab = sorted(
        p.name
        for p in (repo / "examples/prefab/harts").iterdir()
        if (p / "configs/default.toml").is_file()
    )
    designs = sorted(
        str(p.relative_to(repo / "examples/designs"))
        for p in (repo / "examples/designs").glob("*/harts/*")
        if (p / "configs/default.toml").is_file()
    )
    return prefab + designs


async def run(ctx, topic: str, data: dict) -> tuple[str, dict]:
    """Run one step as its own task and wait for its terminal state."""
    trace = str(uuid.uuid4())
    await ctx.enqueue({"topic": topic, "data": {**data, "_trace_id": trace}})
    while True:
        for state in ("success", "failure", "cancelled"):
            value = await ctx.state.get(trace, state)
            if value:
                return state, value.get("body", value)
        await asyncio.sleep(10)


def ball_config(generated: dict) -> dict:
    """What a snapshot keeps of the Ball's default config: no funct7, which
    each design assigns, and no source files."""
    (ball,) = generated["ball"]["balldomain"]["balls"]
    param = {k: v for k, v in ball["param"].items() if k != "_file"}
    host = {k: v for k, v in generated["ball"]["host"].items() if k != "_file"}
    return {
        "name": ball["name"],
        "instructions": [i["mnemonic"] for i in ball["instructions"]],
        "port": ball["port"],
        "param": param,
        "host": host,
    }


def scores(path: Path) -> list[dict]:
    """The rows of an urg table whose header ends in NAME: {name, scores}."""
    lines = path.read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.split()[-1:] == ["NAME"])
    keys = lines[header].split()[:-1]
    rows = []
    for line in lines[header + 1 :]:
        fields = line.split()
        if len(fields) == len(keys) + 1:
            rows.append({"name": fields[-1], "scores": dict(zip(keys, fields[:-1]))})
    return rows


def modules(coverage: Path) -> list[dict]:
    """Coverage per design module, from urg's modlist.txt; tb_top and the UVM
    library are the testbench."""
    return [
        {"module": r["name"], "coverage": r["scores"]}
        for r in scores(coverage / "modlist.txt")
        if r["name"] not in ("tb_top", "uvm_pkg")
    ]


def groups(coverage: Path) -> list[dict]:
    """Covergroup scores, from urg's groups.txt: its columns hold spaces, so
    a row is its first field (the score) and its last (the name)."""
    # urg writes no groups.txt for a testbench without covergroups.
    if not (coverage / "groups.txt").is_file():
        return []
    lines = (coverage / "groups.txt").read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.split()[-1:] == ["NAME"])
    return [
        {"group": line.split()[-1], "score": line.split()[0]}
        for line in lines[header + 1 :]
        if line.strip()
    ]


def hierarchy(area_rpt: str) -> list[dict]:
    """The hierarchical area distribution of DC's area report."""
    lines = area_rpt.splitlines()
    start = next(
        i
        for i, line in enumerate(lines)
        if line.startswith("Hierarchical area distribution")
    )
    rows = []
    for line in lines[start:]:
        fields = line.split()
        if len(fields) == 7 and re.fullmatch(r"[\d.]+", fields[1]):
            instance, total, _percent, comb, noncomb, _black, module = fields
            rows.append(
                {
                    "instance": instance,
                    "module": module,
                    "um2": float(total),
                    "combinational_um2": float(comb),
                    "noncombinational_um2": float(noncomb),
                }
            )
    return rows


def timing(timing_rpt: str) -> dict:
    """The worst setup path of DC's max timing report."""
    start = re.search(r"^\s*Startpoint: (\S+)", timing_rpt, re.M).group(1)
    end = re.search(r"^\s*Endpoint: (\S+)", timing_rpt, re.M).group(1)
    slack = re.search(r"^\s*slack \(\w+\)\s+(-?[\d.]+)", timing_rpt, re.M).group(1)
    return {"slack_ns": float(slack), "startpoint": start, "endpoint": end}


def violations(constraint_rpt: str) -> dict[str, int]:
    """Violated endpoints per constraint kind of DC's constraint report."""
    counts: dict[str, int] = {}
    kind = None
    for line in constraint_rpt.splitlines():
        header = re.match(
            r"^\s+(max_delay/setup|min_delay/hold|max_transition|max_capacitance|max_fanout)\b",
            line,
        )
        if header:
            kind = header.group(1)
            counts[kind] = 0
        elif kind and "(VIOLATED)" in line:
            counts[kind] += 1
    return counts


def power_rows(power_rpt: str) -> list[dict]:
    """The rows of PrimeTime's hierarchical power report, top first, in
    watts. Rows indent two spaces per level; a long name stands on a line of
    its own and its numbers on the next."""
    lines = power_rpt.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("-----")) + 1
    stack: list[str] = []
    rows = []
    i = start
    while i < len(lines):
        line = lines[i]
        fields = line.split()
        if len(fields) >= 6:
            values = fields
        elif fields and i + 1 < len(lines) and len(lines[i + 1].split()) == 5:
            i += 1
            values = lines[i].split()
        else:
            break
        i += 1
        depth = (len(line) - len(line.lstrip())) // 2
        name = fields[0]
        module = fields[1].strip("()") if fields[1].startswith("(") else name
        internal, switching, leakage, total = (float(v) for v in values[-5:-1])
        del stack[depth:]
        stack.append(name)
        rows.append(
            {
                "instance": "/".join(stack[1:]) or stack[0],
                "module": module,
                "internal_w": internal,
                "switching_w": switching,
                "leakage_w": leakage,
                "total_w": total,
            }
        )
    return rows


def area_entry(report_path: str) -> dict:
    (own,) = json.loads(Path(report_path).read_text())["area"]["balls"]
    return {**dc_area(report_path), "ball_um2": own["cell_um2"]}


def unit_area_entry(report_path: str) -> dict:
    """A thread's or a hart's area: the whole top DC synthesized."""
    report = json.loads(Path(report_path).read_text())["area"]
    return {**dc_area(report_path), "um2": report["cell_um2"]}


def dc_area(report_path: str) -> dict:
    report = json.loads(Path(report_path).read_text())["area"]
    reports = Path(report_path).parent.parent / "reports"
    return {
        "status": "pass",
        "constraints_met": report["constraints_met"],
        "hierarchy": hierarchy((reports / "area.rpt").read_text()),
        "timing": timing((reports / "timing_max.rpt").read_text()),
        "violations": violations((reports / "constraint.rpt").read_text()),
        # "DC · <library> · library SHA256 <hash> · SDC <hash>"
        "tool": report["conditions"].split(" · ")[0],
        "library": report["conditions"].split(" · ")[1],
    }


def uvm_entry(uvm: dict) -> dict:
    return {
        "status": "pass",
        "coverage": uvm["coverage"],
        "modules": modules(Path(uvm["log"]) / "coverage"),
        "groups": groups(Path(uvm["log"]) / "coverage"),
    }


def performance_entry(run: dict) -> dict:
    return {"status": "pass", **run["performance"]}


def annotated(activity_rpt: str) -> dict:
    """The share of registers and of all nets whose toggles PrimeTime took
    from the simulation activity: the rest it propagated or defaulted, so a
    low share marks a power figure as an estimate."""
    share = {}
    for line in activity_rpt.splitlines():
        fields = line.split()
        key = {"Sequential": "registers", "Nets": "nets"}.get(
            fields[0] if fields else ""
        )
        if key and key not in share:
            share[key] = float(re.search(r"\(([\d.]+)%\)", fields[1])[1])
    return share


def power_entry(report_path: str) -> dict:
    """A Ball's, a thread's or a hart's power: PT reports from its instance,
    so the first row is it and every other row is under it."""
    reports = Path(report_path).parent.parent / "power-reports"
    rows = power_rows((reports / "power_hierarchy.rpt").read_text())
    return {
        "status": "pass",
        "internal_w": rows[0]["internal_w"],
        "switching_w": rows[0]["switching_w"],
        "leakage_w": rows[0]["leakage_w"],
        "total_w": rows[0]["total_w"],
        "annotated": annotated((reports / "switching_activity.rpt").read_text()),
        "hierarchy": rows,
    }


def listed(body: dict) -> list[str]:
    """The tests of a batch's regression list."""
    return tomllib.loads(Path(body["workload_toml"]).read_text())["workloads"]["tests"]


def suite(body: dict) -> dict:
    """One backend's batch: every listed test with the status its
    summary.json holds; a test the run never reached has none."""
    ran = {
        s["workload_name"]: s["status"]
        for s in (
            json.loads(p.read_text())
            for p in Path(body["log_dir"]).glob("*/summary.json")
        )
    }
    tests = [{"test": t, "status": ran.get(t, "not run")} for t in listed(body)]
    passed = sum(t["status"] == "pass" for t in tests)
    return {
        # A suite that ran nothing proves nothing.
        "status": "pass" if 0 < passed == len(tests) else "failure",
        "log": Path(body["log_dir"]).name,
        "passed": passed,
        "total": len(tests),
        "tests": tests,
    }


async def prepare(ctx, flow: dict, steps: list[tuple[str, dict]]) -> dict | None:
    """Run the steps an evaluation needs, in order, each a topic with what it
    takes besides the flow; the first that fails is the evaluation's failure."""
    for topic, extra in steps:
        state, body = await run(ctx, topic, {**flow, **extra})
        if state != "success":
            return {"status": state, "error": f"{topic}: {body.get('error', state)}"}
    return None


async def unit(ctx, repo: Path, kind: str, name: str, wanted: set, gate) -> dict:
    """A thread or a hart alone, from its default config: its tests (the
    bare-tests regressions on BEMU and Verilator), DC area and power."""
    target = {kind: name}
    selection = require_target(target, (kind,))
    root = repo / "examples" / target["design"]
    entry = {kind: name, "path": str(root.relative_to(repo))}
    flow = {"design": target["design"], "target": selection}
    async with gate:
        # The workloads, and the RTL the Verilator runner and the power
        # simulation are built from.
        failed = None
        if wanted & {"test", "power"}:
            failed = await prepare(
                ctx,
                flow,
                [("workload.build", {}), ("bebop.verilator.verilog", {})]
                + ([("bebop.verilator.build", {})] if "test" in wanted else []),
            )
        if "test" in wanted:
            if failed:
                entry["test"] = failed
            else:
                backends = {}
                for backend, topic, jobs in (
                    ("bemu", "bebop.bemu.batch", 8),
                    ("verilator", "bebop.verilator.batch", 4),
                ):
                    state, body = await run(
                        ctx, topic, {**flow, "test": "bare-tests", "jobs": jobs}
                    )
                    # A batch that ran reports its log directory, passing or not.
                    backends[backend] = (
                        parsed(suite, body)
                        if "log_dir" in body
                        else {
                            "status": state,
                            "error": f"bare-tests: {body.get('error', state)}",
                        }
                    )
                entry["test"] = {
                    "status": (
                        "pass"
                        if all(b["status"] == "pass" for b in backends.values())
                        else "failure"
                    ),
                    "backends": backends,
                }
        if "area" in wanted:
            # `bbdev dc --area`.
            state, body = await run(
                ctx, "dc.verilog", {**flow, "top": None, "from_area_workflow": True}
            )
            entry["area"] = (
                parsed(unit_area_entry, body["report_path"])
                if state == "success"
                else {"status": state, "error": body.get("error", "")}
            )
        if "power" in wanted:
            if failed:
                entry["power"] = failed
                return entry
            # `bbdev dc --power --workload <w>`, with the workload the
            # component declares in power.toml beside its regression lists.
            declared = (
                repo / simulation_target(repo, flow["design"], selection).regression
            ) / "power.toml"
            if not declared.is_file():
                entry["power"] = {
                    "status": "failure",
                    "error": f"no power workload declared: {declared.relative_to(repo)}",
                }
                return entry
            state, body = await run(
                ctx,
                "dc.verilog",
                {
                    **flow,
                    "top": None,
                    "from_power_workflow": True,
                    "workload": tomllib.loads(declared.read_text())["power"][
                        "workload"
                    ],
                },
            )
            # Its DC run is an area run too, when it got that far.
            if "report_path" in body:
                entry["area"] = parsed(unit_area_entry, body["report_path"])
            entry["power"] = (
                parsed(power_entry, body["report_path"])
                if state == "success"
                else {"status": state, "error": body.get("error", "")}
            )
    return entry


def parsed(make, *args) -> dict:
    """An evaluation's entry from its reports. A report that does not parse
    fails that evaluation of that component, not the whole snapshot."""
    try:
        return make(*args)
    except (ValueError, KeyError, OSError, StopIteration) as error:
        return {"status": "failure", "error": f"{make.__name__}: {error!r}"}


async def ball(
    ctx,
    repo: Path,
    name: str,
    area: bool,
    uvm: bool,
    performance: bool,
    power: bool,
    gate,
) -> dict:
    target = {"ball": name}
    selection = require_target(target, ("ball",))
    owner = repo / "examples" / target["design"]
    generated = owner / "output/sims" / selection / "generated/config/config.json"
    entry = {
        "ball": name,
        "path": str(owner.relative_to(repo)),
        "config": ball_config(json.loads(generated.read_text())),
    }
    flow = {"design": target["design"], "top": None, "target": selection}
    if power:
        # `bbdev dc --power --ball`: area, the UVM test's activity, then PT.
        async with gate:
            state, body = await run(
                ctx, "dc.verilog", {**flow, "from_power_workflow": True}
            )
        failed = {"status": state, "error": body.get("error", "")}
        entry["area"] = (
            parsed(area_entry, body["report_path"]) if "report_path" in body else failed
        )
        entry["uvm"] = parsed(uvm_entry, body["uvm"]) if "uvm" in body else failed
        entry["power"] = (
            parsed(power_entry, body["report_path"]) if state == "success" else failed
        )
        return entry
    if area:
        # `bbdev dc --area --ball`.
        async with gate:
            state, body = await run(
                ctx, "dc.verilog", {**flow, "from_area_workflow": True}
            )
        entry["area"] = (
            parsed(area_entry, body["report_path"])
            if state == "success"
            else {"status": state, "error": body.get("error", "")}
        )
    if uvm:
        # `bbdev uvm --run --ball`.
        async with gate:
            state, body = await run(
                ctx,
                "uvm.run",
                {"ball": name, "design": target["design"], "target": selection},
            )
        entry["uvm"] = (
            parsed(uvm_entry, body)
            if state == "success"
            else {"status": state, "error": body.get("error", "")}
        )
    if performance:
        # `bbdev uvm --run '--ball <ball> --performance'`.
        async with gate:
            state, body = await run(
                ctx,
                "uvm.run",
                {
                    "ball": name,
                    "design": target["design"],
                    "target": selection,
                    "performance": True,
                },
            )
        entry["performance"] = (
            parsed(performance_entry, body)
            if state == "success"
            else {"status": state, "error": body.get("error", "")}
        )
    return entry


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def register(save: Path, entry: dict) -> None:
    """Add a snapshot to save/index.json; other bench runs write it too."""
    with (save / "index.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index_path = save / "index.json"
        index = (
            json.loads(index_path.read_text())
            if index_path.is_file()
            else {"snapshots": []}
        )
        index["snapshots"].append(entry)
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin = get_origin_trace_id(input_data, ctx)
    repo = Path(get_buckyball_path())
    (kind,) = [k for k in ("ball", "thread", "hart") if k in input_data]
    try:
        every = {"ball": all_balls, "thread": all_threads, "hart": all_harts}[kind]
        names = every(repo) if input_data[kind] == "all" else [input_data[kind]]
        commit = git(repo, "rev-parse", "HEAD")
        dirty = bool(git(repo, "status", "--porcelain", "--untracked-files=no"))
        taken = datetime.now(timezone.utc)
        # save/<kind>/<subject>/<time>-<commit>[-dirty]; the index id is that path.
        snapshot = f"prefab/{kind}/{taken:%Y-%m-%d-%H%M%S}-{commit[:7]}" + (
            "-dirty" if dirty else ""
        )
        save = repo / "thirdparty/bbbench/save"
        directory = save / snapshot
        directory.mkdir(parents=True, exist_ok=False)
        # The snapshot fills as each component finishes: a run that stops
        # keeps the ones it finished, and the site shows them while it runs.
        done: list[dict] = []

        def write() -> None:
            content = {
                "commit": commit,
                "dirty": dirty,
                "taken": taken.isoformat(),
                f"{kind}s": sorted(done, key=lambda entry: entry["path"]),
            }
            partial = directory / f"{kind}.json.partial"
            partial.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
            partial.replace(directory / f"{kind}.json")

        write()
        register(
            save,
            {
                "id": snapshot,
                "commit": commit,
                "dirty": dirty,
                "taken": taken.isoformat(),
                "files": [f"{kind}.json"],
            },
        )
        gate = asyncio.Semaphore(parallel)
        wanted = {
            k
            for k in ("area", "uvm", "test", "performance", "power")
            if input_data.get(k) is True
        }

        async def one(name: str) -> None:
            try:
                if kind == "ball":
                    entry = await ball(
                        ctx,
                        repo,
                        name,
                        "area" in wanted,
                        "uvm" in wanted,
                        "performance" in wanted,
                        "power" in wanted,
                        gate,
                    )
                else:
                    entry = await unit(ctx, repo, kind, name, wanted, gate)
            except (ValueError, RuntimeError, OSError, KeyError) as error:
                ctx.logger.error(f"bench prefab {name}: {error}")
                return
            done.append(entry)
            write()

        await asyncio.gather(*(one(name) for name in names))
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        await check_result(
            ctx,
            1,
            continue_run=False,
            extra_fields={"task": "prefab", "error": str(error)},
            trace_id=origin,
        )
        return
    await check_result(
        ctx,
        0 if len(done) == len(names) else 1,
        continue_run=False,
        extra_fields={
            "task": "prefab",
            "snapshot": str(directory.relative_to(repo)),
            f"{kind}s": len(done),
        },
        trace_id=origin,
    )
