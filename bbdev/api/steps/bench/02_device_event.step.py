import asyncio
import fcntl
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from motia import FlowContext, queue

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from steps.gvn.scripts.fpga import bitstream_of, fpga_bench
from utils.run_step import run
from utils.event_common import check_result, get_origin_trace_id, require_target
from utils.verilator_turn import verilator_turn
from utils.path import (
    bebop_bin_path,
    design_output_root,
    get_buckyball_path,
    load_design_config,
    log_dir,
)

config = {
    "name": "bench-device",
    "description": "run one step of a device's evaluation timeline into a bbbench snapshot",
    "flows": ["bench"],
    "triggers": [queue("bench.device")],
    "enqueues": [
        "stack.build",
        "workload.build",
        "image.build",
        "bebop.bemu.batch",
        "model.build",
        "bebop.bemu.sim",
        "bebop.verilator.clean",
        "bebop.verilator.verilog",
        "bebop.verilator.build",
        "bebop.verilator.batch",
        "bebop.p2e.verilog",
        "bebop.p2e.buildbitstream",
        "bebop.p2e.batch",
        "workload.tohex",
        "gvn.verilator.perform",
        "gvn.bemu.debug",
        "bebop.p2e.runworkload",
        "uvm.run",
        "dc.verilog",
        "common.task.cancel",
    ],
}

# The evaluation timeline of a design: each node starts once every node it
# needs has passed, whatever else is still running. Faster, rougher levels
# come first, so a wrong design is caught before the slow ones (Verilator,
# DC, FPGA) are spent on it. The Verilator GVN estimate measures performance,
# not correctness: nothing waits for it. Every node runs on the device's one
# default config.
timeline = [
    {"id": "workload", "name": "Workload build", "needs": []},
    {"id": "bemu-workload", "name": "BEMU workload", "needs": ["workload"]},
    {"id": "model-build", "name": "Model build", "needs": ["workload"]},
    {"id": "bench-build", "name": "Bench build", "needs": ["workload"]},
    {
        "id": "verilator-workload",
        "name": "Verilator workload",
        "needs": ["bemu-workload"],
    },
    {
        "id": "bemu-model",
        "name": "BEMU model",
        "needs": ["bemu-workload", "model-build"],
    },
    {
        "id": "bemu-bench",
        "name": "BEMU bench",
        "needs": ["bemu-workload", "bench-build"],
    },
    {
        "id": "bemu-accuracy",
        "name": "BEMU accuracy",
        "needs": ["bemu-workload", "model-build"],
    },
    {
        "id": "verilator-gvn",
        "name": "Verilator GVN",
        "needs": ["verilator-workload", "model-build"],
    },
    {"id": "dc-area", "name": "DC area", "needs": ["verilator-workload"]},
    {"id": "power", "name": "Power", "needs": ["dc-area"]},
    {"id": "uvm", "name": "UVM", "needs": ["verilator-workload"]},
    {"id": "fpga-bitstream", "name": "FPGA bitstream", "needs": ["verilator-workload"]},
    {"id": "fpga-workload", "name": "FPGA workload", "needs": ["fpga-bitstream"]},
    {"id": "fpga-gvn", "name": "FPGA GVN", "needs": ["fpga-workload"]},
    {"id": "fpga-model", "name": "FPGA model", "needs": ["fpga-gvn"]},
]


# No simulation runs longer than this; one that would is cut into parts or
# its cause fixed, never waited out. Batches need no limit of their own: the
# regression harness stops each of their simulations at the same 30 minutes.
simulation_limit = 30 * 60


def hart_cycles(states: list[Path]) -> int:
    """BEMU's simulated cycles: the latest elapsed_cycles each hart's
    cpu-state.log holds, the slowest hart's."""
    return max(
        int(re.search(r"elapsed_cycles=(\d+)", s.read_text()).group(1)) for s in states
    )


def cycles(workload_dir: Path) -> int | None:
    """Simulated cycles of a batch workload: BEMU's from its cpu-state.log,
    Verilator's the harness clock count it prints at its end. A simulation
    stopped before its end leaves none."""
    states = sorted(workload_dir.glob("log/id-*/cpu-state.log"))
    if states:
        return hart_cycles(states)
    found = re.search(
        r"^Simulation cycles: (\d+)$", (workload_dir / "stdout.log").read_text(), re.M
    )
    return int(found.group(1)) if found else None


def suite(log_dir: Path) -> dict:
    """A batch run's workloads, from the summary.json each one leaves."""
    workloads = []
    for summary in sorted(log_dir.glob("*/summary.json")):
        s = json.loads(summary.read_text())
        workload = {
            "workload": s["workload_name"],
            "status": s["status"],
            "elapsed_s": float(s["elapsed_sec"].removesuffix("s")),
        }
        simulated = cycles(summary.parent)
        if simulated is not None:
            workload["cycles"] = simulated
        workloads.append(workload)
    return {
        "log": log_dir.name,
        "passed": sum(w["status"] == "pass" for w in workloads),
        "total": len(workloads),
        "workloads": workloads,
    }


async def build(ctx, target: dict) -> dict:
    # Its stack and its workloads; models and benchmarks build in their steps.
    for topic in ("stack.build", "workload.build"):
        state, body = await run(ctx, topic, target)
        if state != "success":
            return {
                "status": "failure",
                "error": f"{topic}: {body.get('detail', body.get('error', state))}",
            }
    return {"status": "pass"}


async def bemu_batch(ctx, target: dict) -> dict:
    suites = {}
    for test in ("bare-tests", "linux-tests"):
        if test == "linux-tests":
            state, body = await run(ctx, "image.build", target)
            if state != "success":
                return {
                    "status": "failure",
                    "error": f"image.build: {body.get('error', state)}",
                    "suites": suites,
                }
        state, body = await run(
            ctx, "bebop.bemu.batch", {**target, "test": test, "jobs": 16}
        )
        # A batch that ran reports its log directory, passing or not; a
        # build failure has none.
        if "log_dir" not in body:
            return {
                "status": "failure",
                "error": f"{test}: {body.get('error', state)}",
                "suites": suites,
            }
        suites[test] = suite(Path(body["log_dir"]))
    # A suite that ran nothing proves nothing.
    passed = all(0 < s["passed"] == s["total"] for s in suites.values())
    return {"status": "pass" if passed else "failure", "suites": suites}


async def verilator_batch(ctx, target: dict) -> dict:
    # The stack and workloads of the config are the build node's; here the CI
    # sequence: clean, verilog and build with DiffTest, the bare workloads.
    # The designs walking in parallel take turns at their Verilator build and
    # regression: each takes many cores, and the host threads of a model spin
    # while they wait, so regressions side by side stop each other.
    # The turns go first come, first served (utils/verilator_turn.py).
    turn = verilator_turn(Path(get_buckyball_path()))
    await asyncio.to_thread(turn.__enter__)
    try:
        for topic, extra in (
            ("bebop.verilator.clean", {}),
            ("bebop.verilator.verilog", {"diff": True}),
            ("bebop.verilator.build", {"diff": True, "jobs": 100}),
        ):
            state, body = await run(ctx, topic, {**target, **extra})
            if state != "success":
                return {
                    "status": "failure",
                    "error": f"{topic}: {body.get('error', state)}",
                }
        state, body = await run(
            ctx,
            "bebop.verilator.batch",
            # The runner sizes its workers to half the host over the model's
            # host threads.
            {**target, "test": "bare-tests", "diff": True, "jobs": 16},
        )
    finally:
        turn.__exit__(None, None, None)
    if "log_dir" not in body:
        return {"status": "failure", "error": f"bare-tests: {body.get('error', state)}"}
    s = suite(Path(body["log_dir"]))
    return {
        "status": "pass" if 0 < s["passed"] == s["total"] else "failure",
        "suites": {"bare-tests": s},
    }


def models(target: dict) -> list[str]:
    """The target models of the design, from bbmap."""
    (entry,) = [
        d
        for d in json.loads(
            (
                Path(get_buckyball_path()) / "thirdparty/bbbench/save/bbmap.json"
            ).read_text()
        )["designs"]
        if d["design"] == target["design"]
    ]
    return entry["models"]


async def model_build(ctx, target: dict) -> dict:
    """Every model recipe of the design, built (`model --build --model all`).
    A design without target models (a CPU system measured by its benchmarks,
    like multi-rocket) skips this node."""
    if not models(target):
        return {"status": "skipped"}
    state, body = await run(ctx, "model.build", {**target, "model": "all"})
    if state != "success":
        return {
            "status": "failure",
            "error": f"model.build: {body.get('detail', body.get('error', state))}",
        }
    return {"status": "pass"}


async def bemu_model(ctx, target: dict) -> dict:
    """Every target model of the design (from bbmap), built by the model build
    node, run whole on BEMU, bare metal and under Linux, one simulation each
    within the limit. A model passes when its program exits 0, which its own
    check against the reference output decides; its inference cycles and
    output difference come from its report lines."""
    if not models(target):
        return {"status": "skipped"}
    results = []
    for model in models(target):
        # Bare metal, then under Linux: the image carrying the model, whose
        # init runs it instead of the regression.
        for os_name in ("baremetal", "linux"):
            if os_name == "linux":
                state, body = await run(ctx, "image.build", {**target, "model": model})
                if state != "success":
                    results.append(
                        {
                            "model": model,
                            "os": os_name,
                            "status": "failure",
                            "error": f"image.build: {body.get('detail', body.get('error', state))}",
                        }
                    )
                    continue
            state, body = await run(
                ctx,
                "bebop.bemu.sim",
                {**target, "model": model, "os": os_name},
                simulation_limit,
            )
            result = {
                "model": model,
                "os": os_name,
                "status": {"success": "pass", "timeout": "timeout"}.get(
                    state, "failure"
                ),
            }
            if "log_dir" in body:
                text = (Path(body["log_dir"]) / "run.log").read_text()
                result["log"] = Path(body["log_dir"]).name
                # The whole simulation, boot to exit, beside the inference alone.
                result["simulated_cycles"] = hart_cycles(
                    sorted(Path(body["log_dir"]).glob("id-*/cpu-state.log"))
                )
                inference = re.search(r"Inference cycles: (\d+)", text)
                if inference:
                    result["cycles"] = int(inference.group(1))
                # The program's own check against its reference output (stack/models/lib/testutils.h).
                check = re.search(
                    r"Output max diff: ([0-9.eE+-]+) \((\d+) elements, relative ([0-9.eE+-]+)\)",
                    text,
                )
                if check:
                    result["max_diff"] = float(check.group(1))
                    result["elements"] = int(check.group(2))
                    result["relative"] = float(check.group(3))
                wall = re.search(r"BEMU wall time: ([\d.]+)s", text)
                if wall:
                    result["elapsed_s"] = float(wall.group(1))
            else:
                result["error"] = body.get("error", state)
            results.append(result)
    return {
        "status": "pass" if all(m["status"] == "pass" for m in results) else "failure",
        "models": results,
    }


def benches(target: dict) -> list[str]:
    """The benchmarks the device lists (sims/device/regression/bemu/bench.toml,
    of bb-tests/workloads/src/benchmark); none without the file."""
    listed = (
        Path(get_buckyball_path())
        / "examples/designs"
        / target["design"]
        / "devices"
        / target["device"]
        / "sims/device/regression/bemu/bench.toml"
    )
    if not listed.is_file():
        return []
    return tomllib.loads(listed.read_text())["bench"]["benches"]


async def bench_build(ctx, target: dict) -> dict:
    """The benchmarks the device lists, built (`bench --build`); a device
    without benchmarks skips this node."""
    if not benches(target):
        return {"status": "skipped"}
    for bench in benches(target):
        state, body = await run(ctx, "workload.build", {**target, "bench": bench})
        if state != "success":
            return {
                "status": "failure",
                "error": f"bench build {bench}: {body.get('detail', body.get('error', state))}",
            }
    return {"status": "pass"}


async def bemu_bench(ctx, target: dict) -> dict:
    """The benchmarks the device lists, built by the bench build node, run on
    BEMU bare metal and under Linux, the whole suite in one simulation each
    within the limit. A benchmark passes when its program exits 0, which its
    own checks of each result decide; its cycles are its own count of the
    measured part (embench's total, coremark's ticks)."""
    if not benches(target):
        return {"status": "skipped"}
    results = []
    for bench in benches(target):
        # Bare metal, then under Linux: the image whose init runs the
        # benchmark's Linux program instead of the regression.
        for os_name in ("baremetal", "linux"):
            if os_name == "linux":
                state, body = await run(ctx, "image.build", {**target, "bench": bench})
                if state != "success":
                    results.append(
                        {
                            "bench": bench,
                            "os": os_name,
                            "status": "failure",
                            "error": f"image.build: {body.get('detail', body.get('error', state))}",
                        }
                    )
                    continue
            state, body = await run(
                ctx,
                "bebop.bemu.sim",
                {**target, "bench": bench, "os": os_name},
                simulation_limit,
            )
            result = {
                "bench": bench,
                "os": os_name,
                "status": {"success": "pass", "timeout": "timeout"}.get(
                    state, "failure"
                ),
            }
            if "log_dir" in body:
                text = (Path(body["log_dir"]) / "run.log").read_text()
                result["log"] = Path(body["log_dir"]).name
                result["simulated_cycles"] = hart_cycles(
                    sorted(Path(body["log_dir"]).glob("id-*/cpu-state.log"))
                )
                cycles = re.search(
                    r"Embench top total cycles=(\d+)|Total ticks\s*:\s*(\d+)", text
                )
                if cycles:
                    result["cycles"] = int(cycles.group(1) or cycles.group(2))
                wall = re.search(r"BEMU wall time: ([\d.]+)s", text)
                if wall:
                    result["elapsed_s"] = float(wall.group(1))
            else:
                result["error"] = body.get("error", state)
            results.append(result)
    return {
        "status": "pass" if all(b["status"] == "pass" for b in results) else "failure",
        "benches": results,
    }


async def bemu_accuracy(ctx, target: dict) -> dict:
    """The accuracy of every target model with a dataset: the model build
    writes models/<model>/dataset/ (files of one record per sample, as many
    as labels.data holds int64s; under model/, files every run needs whole,
    such as weights the program reads from the region) and the program
    <design>-<model>-dataset, which runs the samples it finds in the model
    region of the DDR. The samples are cut into shards that fit the region
    beside model/, at least 4096 of them so each run stays short, as many
    BEMU runs at a time as the host has CPUs for (at most 64), each loading its shard packed into
    <program>-ddr.bin. Per model: the task
    accuracy (samples classified as labelled), the pass rate (samples
    predicting what the reference predicts) and the accuracy of the
    reference (PyTorch) on the same samples; for a model of continuous
    outputs, the mean squared errors of the device's and the reference's
    outputs against the recorded ones. A design without datasets skips
    this node."""
    repo = get_buckyball_path()
    design, selection = target["design"], target["target"]
    root = Path(design_output_root(repo, design, selection=selection))
    # One accuracy run at a time on the host: each holds 64 BEMUs. The
    # programs are those built when the run gets the host.
    with open(Path(repo) / "log/bench-accuracy.lock", "w") as lock:
        await asyncio.to_thread(fcntl.flock, lock, fcntl.LOCK_EX)
        programs = {
            model: root / "workloads/models" / f"{design}-{model}-dataset"
            for model in models(target)
            if (root / "workloads/models" / f"{design}-{model}-dataset").is_file()
        }
        if not programs:
            return {"status": "skipped"}
        bemu = bebop_bin_path(repo, design, "bemu", selection=selection)
        ddr = load_design_config(repo, design, selection=selection).device.ddr
        results = []
        for model in sorted(
            programs,
            key=lambda m: (root / "models" / m / "dataset/labels.data").stat().st_size,
        ):
            program = programs[model]
            data = root / "models" / model / "dataset"
            pack = (
                Path(repo)
                / "stack/models/build"
                / design
                / "device"
                / target["device"]
                / Path(target["config"]).stem
                / model
                / "model-pack"
            )
            files = sorted(path.name for path in data.iterdir() if path.is_file())
            whole = (
                sorted(
                    str(path.relative_to(data / "model"))
                    for path in (data / "model").rglob("*")
                    if path.is_file()
                )
                if (data / "model").is_dir()
                else []
            )
            samples = (data / "labels.data").stat().st_size // 8
            records = {name: (data / name).stat().st_size // samples for name in files}
            # A shard fits the model region beside the whole files (each file at
            # a 4 KiB boundary after the header page); there are at least 64.
            room = (
                ddr.model
                - 4096 * (len(files) + len(whole) + 1)
                - sum(
                    -(-(data / "model" / name).stat().st_size // 4096) * 4096
                    for name in whole
                )
            )
            count = min(-(-samples // 4096), room // sum(records.values()))
            logs = Path(
                log_dir(
                    repo,
                    design,
                    "device",
                    datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                    "accuracy",
                    model,
                    selection=selection,
                )
            )
            logs.mkdir(parents=True)
            # All shards run the same ELF even if a new build replaces the installed program.
            program = Path(shutil.copy2(program, logs / "program"))

            async def shard(first: int, taken: int) -> tuple[int, str]:
                async with slots:
                    log = logs / f"shard{first}"
                    (log / "data").mkdir(parents=True)
                    for name in files:
                        with (data / name).open("rb") as source:
                            source.seek(first * records[name])
                            (log / "data" / name).write_bytes(
                                source.read(taken * records[name])
                            )
                    for name in whole:
                        (log / "data" / name).parent.mkdir(parents=True, exist_ok=True)
                        (log / "data" / name).symlink_to(
                            (data / "model" / name).resolve()
                        )
                    await (
                        await asyncio.create_subprocess_exec(
                            str(pack),
                            str(log / "model.img"),
                            str(log / "data"),
                            *files,
                            *whole,
                        )
                    ).wait()
                    # The DDR image: the shard's model image in the model region.
                    with (log / "program-ddr.bin").open("wb") as image:
                        image.truncate(ddr.bytes)
                        image.seek(ddr.bytes - ddr.model)
                        image.write((log / "model.img").read_bytes())
                    # The image (whole files such as weights take GBs) is in the
                    # DDR image now.
                    (log / "model.img").unlink()
                    run = await asyncio.create_subprocess_exec(
                        bemu,
                        "--elf",
                        str(program),
                        "--ddr-image",
                        str(log / "program-ddr.bin"),
                        "--log-dir",
                        str(log),
                        cwd=log,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                    )
                    try:
                        output, _ = await asyncio.wait_for(
                            run.communicate(), simulation_limit
                        )
                    except TimeoutError:
                        # A shard past the limit is a failed shard; its BEMU
                        # must not keep the host.
                        run.kill()
                        output, _ = await run.communicate()
                    (log / "run.log").write_bytes(output)
                    (log / "program-ddr.bin").unlink()
                    shutil.rmtree(log / "data")
                    return run.returncode, output.decode(errors="replace")

            start = time.monotonic()
            # Include the timed first sample in the result. Leave room for host
            # contention when the rest of the shards run concurrently: as many
            # BEMUs as the host has CPUs for the host threads one of them
            # keeps busy, at most 64.
            slots = asyncio.Semaphore(1)
            used = resource.getrusage(resource.RUSAGE_CHILDREN)
            first_run = await shard(0, 1)
            runs = [first_run]
            if first_run[0] == 0:
                seconds = time.monotonic() - start
                now = resource.getrusage(resource.RUSAGE_CHILDREN)
                threads = (
                    now.ru_utime + now.ru_stime - used.ru_utime - used.ru_stime
                ) / seconds
                slots = asyncio.Semaphore(
                    min(64, max(1, int(os.cpu_count() / max(1.0, threads))))
                )
                count = min(count, max(1, int((simulation_limit / 8) / seconds)))
                runs.extend(
                    await asyncio.gather(
                        *(
                            shard(first, min(count, samples - first))
                            for first in range(1, samples, count)
                        )
                    )
                )
            totals = [0, 0, 0, 0]
            # A model of continuous outputs adds the squared errors of the
            # device's and the reference's against the recorded outputs.
            errors = [0.0, 0.0, 0]
            failed = 0
            for code, output in runs:
                line = re.search(
                    r"Dataset: (\d+) samples, (\d+) correct, (\d+) passed,"
                    r" (\d+) reference correct",
                    output,
                )
                if code or not line:
                    failed += 1
                    continue
                totals = [a + int(b) for a, b in zip(totals, line.groups())]
                error = re.search(r"Squared error: (\S+) (\S+) (\d+)", output)
                if error:
                    errors = [
                        errors[0] + float(error[1]),
                        errors[1] + float(error[2]),
                        errors[2] + int(error[3]),
                    ]
            ran, correct, passed, golden = totals
            result = {
                "model": model,
                "status": "pass" if not failed and ran == samples else "failure",
                "samples": samples,
                "ran": ran,
                "accuracy": correct / samples,
                "pass_rate": passed / samples,
                "reference_accuracy": golden / samples,
                "elapsed_s": round(time.monotonic() - start, 1),
                "log": logs.name,
            }
            if errors[2]:
                result["action_mse"] = errors[0] / errors[2]
                result["reference_action_mse"] = errors[1] / errors[2]
            if failed:
                result["error"] = f"{failed} of {len(runs)} shards failed"
            results.append(result)
        return {
            "status": (
                "pass" if all(r["status"] == "pass" for r in results) else "failure"
            ),
            "datasets": results,
        }


async def verilator_gvn(ctx, target: dict) -> dict:
    """A rough end-to-end NPU cycle count of every target model on Verilator
    (`gvn-verilator --perform`): each distinct GVN slice once, times the times it
    runs; per slice, its repeats, its Verilator cycles and its share. A
    design without models or without a GVN driver skips this node."""
    driver = (
        Path(get_buckyball_path())
        / "examples/designs"
        / target["design"]
        / "devices"
        / target["device"]
        / "sims/device/gvn/run.py"
    )
    if not models(target) or not driver.is_file():
        return {"status": "skipped"}
    results = []
    for model in models(target):
        state, body = await run(
            ctx, "gvn.verilator.perform", {**target, "model": model}
        )
        result = {"model": model, "status": "failure"}
        if "estimate" in body and Path(body["estimate"]).is_file():
            estimate = json.loads(Path(body["estimate"]).read_text())
            result["cycles"] = estimate["cycles"]
            result["elapsed_s"] = estimate["seconds"]
            result["slices"] = estimate["slices"]
            if (
                state == "success"
                and result["slices"]
                and all(
                    s["exit"] == 0 and (s["cycles"] or 0) > 0 for s in result["slices"]
                )
            ):
                result["status"] = "pass"
            # What the cycles leave out, apart from what they measure.
            for kind in ("measured", "timed_out", "not_run", "failed"):
                result[kind] = estimate[kind]
            result["unmeasured_bemu_cycles"] = estimate["unmeasured_bemu_cycles"]
            result["log"] = Path(body["estimate"]).parent.name
        else:
            result["error"] = body.get("error") or "GVN estimate was not produced"
        results.append(result)
    return {
        "status": "pass" if all(r["status"] == "pass" for r in results) else "failure",
        "estimates": results,
    }


async def uvm(ctx, target: dict, node: dict, write) -> dict:
    config = load_design_config(
        get_buckyball_path(), target["design"], selection=target["target"]
    )
    runnable = {profile.ip for profile in config.verification.profiles}
    results = [
        {"ip": ip, "status": "pending" if ip in runnable else "unavailable"}
        for ip in config.verification.ips
    ]
    node["uvm"] = results
    if not runnable:
        return {"status": "skipped"}
    for record in results:
        ip = record["ip"]
        if ip not in runnable:
            continue
        record["status"] = "running"
        write()
        state, body = await run(ctx, "uvm.run", {**target, "ip": ip})
        if "results" in body:
            record.update(log=Path(body["log"]).name, profiles=body["results"])
        if state != "success":
            record.update(
                status="failure",
                error="\n".join(body.get("failures", [])) or body.get("error", state),
            )
            continue
        record["status"] = "pass"
    return {
        "status": (
            "failure" if any(r["status"] == "failure" for r in results) else "pass"
        ),
    }


async def fpga_workload(ctx, target: dict, request: dict) -> dict:
    """The bare workloads of the device on its FPGA (P2E), on the bitstream
    the request names or the device's latest completed build, at the board
    location the request names (0.A by default), as the Verilator workload
    node runs them; with DiffTest when the bitstream carries it."""
    bit, diff = bitstream_of(target, request)
    # P2E loads each workload as the hex image of its ELF.
    state, body = await run(ctx, "workload.tohex", target)
    if state != "success":
        return {
            "status": "failure",
            "error": f"workload.tohex: {body.get('error', state)}",
        }
    state, body = await run(
        ctx,
        "bebop.p2e.batch",
        {
            **target,
            "bitstream": str(bit),
            "test": "bare-tests",
            "diff": diff,
            "fpga-location": request.get("fpga-location", "0.A"),
        },
    )
    if "log_dir" not in body:
        return {"status": "failure", "error": f"bare-tests: {body.get('error', state)}"}
    s = suite(Path(body["log_dir"]))
    return {
        "status": "pass" if 0 < s["passed"] == s["total"] else "failure",
        "bitstream": str(bit),
        "diff": diff,
        "suites": {"bare-tests": s},
    }


async def dc_area(ctx, target: dict) -> dict:
    """The device's DC area (`dc --area`): its synthesized cell area and
    whether it met its constraints."""
    state, body = await run(
        ctx, "dc.verilog", {**target, "top": None, "from_area_workflow": True}
    )
    if state != "success":
        return {
            "status": "failure",
            "error": f"dc.area: {body.get('detail', body.get('error', state))}",
        }
    report = json.loads(Path(body["report_path"]).read_text())["area"]
    return {
        "status": "pass",
        "um2": report["cell_um2"],
        "constraints_met": report["constraints_met"],
        "report": body["report_path"],
    }


async def dc_power(ctx, target: dict) -> dict:
    """The device's power (`dc --power`) on the design's power workload: the
    top row of PrimeTime's hierarchical report, in watts."""
    state, body = await run(
        ctx, "dc.verilog", {**target, "top": None, "from_power_workflow": True}
    )
    if state != "success":
        return {
            "status": "failure",
            "error": f"dc.power: {body.get('detail', body.get('error', state))}",
        }
    rpt = (
        Path(body["report_path"]).parent.parent / "power-reports/power_hierarchy.rpt"
    ).read_text()
    lines = rpt.splitlines()
    top = next(i for i, line in enumerate(lines) if line.startswith("-----")) + 1
    values = (
        lines[top].split() if len(lines[top].split()) >= 6 else lines[top + 1].split()
    )
    internal, switching, leakage, total = (float(v) for v in values[-5:-1])
    return {
        "status": "pass",
        "internal_w": internal,
        "switching_w": switching,
        "leakage_w": leakage,
        "total_w": total,
        "report": body["report_path"],
    }


async def buildbitstream(ctx, target: dict) -> dict:
    for topic in ("bebop.p2e.verilog", "bebop.p2e.buildbitstream"):
        state, body = await run(ctx, topic, target)
        if state != "success":
            return {
                "status": "failure",
                "error": f"{topic}: {body.get('detail', body.get('error', state))}",
            }
    for artifact in ("bitstream", "rtcfg", "libvCtb", "runtime"):
        if not Path(body[artifact]).is_file():
            return {
                "status": "failure",
                "error": f"Missing FPGA artifact: {body[artifact]}",
            }
    return {"status": "pass", "bitstream": body["bitstream"], "diff": body["diff"]}


# The nodes this bench runs so far; the rest of the timeline is pending.
steps = {
    "workload": build,
    "bemu-workload": bemu_batch,
    "model-build": model_build,
    "bench-build": bench_build,
    "verilator-workload": verilator_batch,
    "bemu-model": bemu_model,
    "bemu-bench": bemu_bench,
    "bemu-accuracy": bemu_accuracy,
    "verilator-gvn": verilator_gvn,
    "dc-area": dc_area,
    "power": dc_power,
    "fpga-bitstream": buildbitstream,
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
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


async def bench(ctx, repo: Path, device: str, step: str, request: dict) -> dict:
    """The step of the timeline of one device, as its snapshot; the other
    steps stay pending."""
    # The device's one default config, for every node.
    target = {"device": device}
    if step == "fpga-bitstream":
        target["diff"] = request.get("diff", False)
    target["target"] = require_target(target, ("device",))
    design = target["design"]
    commit = git(repo, "rev-parse", "HEAD")
    dirty = bool(git(repo, "status", "--porcelain", "--untracked-files=no"))
    taken = datetime.now(timezone.utc)
    # save/<kind>/<subject>/<time>-<commit>[-dirty]; the index id is that path.
    stamp = taken.strftime("%Y-%m-%d-%H%M%S")
    if step in ("fpga-gvn", "fpga-model"):
        stamp += taken.strftime("-%f")
    snapshot = f"designs/{design}/{stamp}-{commit[:7]}" + ("-dirty" if dirty else "")
    save = repo / "thirdparty/bbbench/save"
    directory = save / snapshot
    directory.mkdir(parents=True, exist_ok=False)
    nodes = [{**node, "status": "pending"} for node in timeline]

    # The snapshot is written before the first node and again as each
    # node starts and ends, so a page shows the walk as it goes.
    def write() -> None:
        temporary = directory / "design.json.tmp"
        temporary.write_text(
            json.dumps(
                {
                    "commit": commit,
                    "dirty": dirty,
                    "taken": taken.isoformat(),
                    "design": design,
                    "config": target["config"],
                    "nodes": nodes,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        temporary.replace(directory / "design.json")

    write()
    register(
        save,
        {
            "id": snapshot,
            "commit": commit,
            "dirty": dirty,
            "taken": taken.isoformat(),
            "files": ["design.json"],
        },
    )

    async def walk(node: dict) -> None:
        start = time.monotonic()
        if node["id"] in ("fpga-model", "fpga-gvn"):
            try:
                selected = models(target)
                if "model" in request:
                    if request["model"] not in selected:
                        raise ValueError(
                            f"Unknown model for {target['device']}: {request['model']}"
                        )
                    selected = [request["model"]]
                node.update(
                    await fpga_bench(ctx, target, node, request, selected, write, False)
                )
            except (
                ValueError,
                RuntimeError,
                OSError,
                KeyError,
                subprocess.SubprocessError,
            ) as error:
                for row in node.get("models", []) + node.get("estimates", []):
                    if row["status"] == "running":
                        row.update(status="failure", error=str(error))
                node.update(status="failure", error=str(error))
        elif node["id"] == "uvm":
            node.update(await uvm(ctx, target, node, write))
        elif node["id"] == "fpga-workload":
            try:
                node.update(await fpga_workload(ctx, target, request))
            except ValueError as error:
                node.update(status="failure", error=str(error))
        else:
            node.update(await steps[node["id"]](ctx, target))
        node["elapsed_s"] = round(time.monotonic() - start, 1)
        write()

    (node,) = [n for n in nodes if n["id"] == step]
    node["status"] = "running"
    write()
    await walk(node)
    return {"snapshot": snapshot, "status": node["status"]}


async def handler(input_data: dict, ctx: FlowContext) -> None:
    origin = get_origin_trace_id(input_data, ctx)
    repo = Path(get_buckyball_path())
    step = input_data["step"]
    devices = (
        sorted(
            p.name
            for p in (repo / "examples/designs").glob("*/devices/*")
            if p.is_dir()
        )
        if input_data["device"] == "all"
        else [input_data["device"]]
    )
    results, errors = {}, {}

    async def evaluate(device: str) -> None:
        try:
            results[device] = await bench(ctx, repo, device, step, input_data)
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            errors[device] = str(error)

    if step == "fpga-bitstream" and input_data["device"] == "all":
        single_fpga = {"pebble", "toy", "goban", "gemmini"}
        await asyncio.gather(
            *(evaluate(device) for device in devices if device in single_fpga)
        )
        devices = [device for device in devices if device not in single_fpga]
    for device in devices:
        await evaluate(device)
    failed = errors or any(r["status"] == "failure" for r in results.values())
    await check_result(
        ctx,
        1 if failed else 0,
        continue_run=False,
        extra_fields={
            "task": "device",
            "step": step,
            "devices": results,
            "errors": errors,
        },
        trace_id=origin,
    )
