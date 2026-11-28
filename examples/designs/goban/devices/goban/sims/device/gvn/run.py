"""GVN for Goban (one hart, its hardware threads): cut every hardware thread
subgraph of the hart graph into slices at its DDR values, take the values
they read and write from the GPU run of the linalg graph, and run each
iteration of each slice (one tile iteration) as its own NPU program on BEMU.

  bbdev gvn-bemu --debug '--config goban/devices/goban/configs/GobanGvnConfig.toml --device goban --model laya'

Inputs: the compiled model of the device config (`bbdev model --build`:
models/<model>/ of the config output and its recipe build) and the GPU
harness gvn/gpu.cpp of the model recipe. Outputs under gvn/<model> of the
config output; BEMU logs under log/.

The model recipe's tree (its gvn/) checks the thread subgraphs of the hart
graph, cuts one slice per check and lowers it; the operators of the checked
graph must equal the compiled model's, and each slice's physical bank
operations must appear, in order, among those of its subgraph in the whole
graph. stack/gvn/gpu computes the values the slices read and write; this
directory's CMakeLists.txt links each slice with its readback harness.
"""

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import hashlib
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[8] / "bbdev/api/steps/gvn/scripts")
)
from gvn import bank_operations, build_plugins, repo, run, table_offset  # noqa: E402
from utils.path import load_design_config, log_dir  # noqa: E402


def function_bank_operations(path, name):
    """The physical bank operations of function `name`."""
    text = path.read_text()
    begin = re.search(rf"func\.func (?:private )?@{name}\(", text).start()
    following = re.search(r"\n  func\.func ", text[begin + 1 :])
    body = text[begin : begin + 1 + following.start()] if following else text[begin:]
    return bank_operations(body)


def named(layout):
    """The GPU values of a slice: its graph values, arguments and output."""
    found = {layout["output"]["id"]}
    for entry in layout["entries"]:
        if entry["kind"] == "value":
            found.add(entry["id"])
        if entry["kind"] == "argument":
            found.add(f"arg:{entry['index']}")
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()
    args.design = Path(args.config).parts[0]
    stem = Path(args.config).stem
    selection = f"device/{args.device}/{stem}"

    config = (
        repo
        / "examples/designs"
        / args.design
        / "output/sims/device"
        / args.device
        / stem
    )
    model = config / "models" / args.model
    harness = repo / "examples/models" / args.design / args.model / "gvn"
    out = config / "gvn" / args.model
    npu = out / "npu"
    out.mkdir(parents=True, exist_ok=True)
    build_plugins()

    # ===--- NPU side: the slices of the thread subgraphs ---=== #
    recipe_build = repo / "stack/models/build" / args.design / selection / args.model
    run(
        ["cmake", "-S", repo / "stack/models", "-B", recipe_build],
        stdout=subprocess.DEVNULL,
    )
    run(["cmake", "--build", recipe_build, "--target", f"{args.model}-gvn-checks"])
    if (npu / "gvn_operators.mlir").read_text() != (
        npu / "model_operators.mlir"
    ).read_text():
        raise SystemExit("slice GP operators differ from the compiled model")
    # A check belongs to the function it is in, as the recipe's tree counts.
    slices = []
    function = None
    for line in (npu / "checks.mlir").read_text().splitlines():
        match = re.match(r"\s*func\.func (?:private )?@(\w+)\(", line)
        if match:
            function, number = match.group(1), 0
        if "gvn.check " in line:
            slices.append(f"{function}.{number}")
            number += 1
    print(f"{len(slices)} slices", flush=True)

    # Debug runs seek coverage: one instance of each distinct slice (its
    # first iteration). Slices of thread graphs with the same body (the same
    # computation on another thread, other rows or another part of a
    # parallel group: the hart's slice code differs only in those) are one;
    # only one of them is compiled. A slice runs once per iteration in the
    # whole graph: the repeats of a distinct slice are its iterations times
    # the slices that are it.
    bodies = {
        function: hashlib.sha256(
            re.sub(
                r" loc\((?:[^()]|\([^()]*\))*\)|hart\.parallel_part = \d+ : i64",
                "",
                body,
            ).encode()
        ).hexdigest()
        for function, body in re.findall(
            r"^  func\.func private @(\w+)\((.*?)^  \}",
            (npu / "hart.mlir").read_text(),
            re.M | re.S,
        )
    }
    patterns, count = {}, {}
    for name in slices:
        # <function>.<part>: the part of the thread graph the slice keeps.
        function, part = name.rsplit(".", 1)
        code = (bodies[function], part)
        patterns.setdefault(code, name)
        count[patterns[code]] = count.get(patterns[code], 0) + 1
    distinct = list(patterns.values())
    print(
        f"{len(distinct)} distinct slices of {len(slices)}: compiling those",
        flush=True,
    )
    run(
        ["cmake", "-S", repo / "stack/models", "-B", recipe_build],
        stdout=subprocess.DEVNULL,
    )
    run(
        [
            "cmake",
            "--build",
            recipe_build,
            "--parallel",
            str(args.jobs),
            "--target",
            f"{args.model}-model-build",
            npu / "whole_physical.mlir",
            npu / "operators.embed.o",
            *(out / name / "slice.o" for name in distinct),
        ]
    )
    layouts = {}
    for name in distinct:
        function = name.rsplit(".", 1)[0]
        operations = function_bank_operations(
            out / name / "slice_physical.mlir", function
        )
        # In order, not contiguous: an iteration of a loop loads the operands
        # the whole graph loads once before that loop.
        remaining = iter(
            function_bank_operations(npu / "whole_physical.mlir", function)
        )
        if not all(operation in remaining for operation in operations):
            raise SystemExit(
                f"{name}: its physical bank operations differ from those of the whole graph"
            )
        layouts[name] = json.loads((out / name / "layout.json").read_text())
    repeats = {name: count[name] * layouts[name]["iterations"] for name in distinct}
    print(
        "physical banks: every distinct slice appears in its subgraph in the whole graph",
        flush=True,
    )

    # ===--- GPU side: the values the slices read and write ---=== #
    ids = sorted(set().union(*(named(layout) for layout in layouts.values())))
    gpu = out / "gpu"
    gpu.mkdir(exist_ok=True)
    (gpu / "ids").write_text(",".join(ids))
    shutil.copy2(model / "subgraph0_linalg.mlir", gpu / "linalg.mlir")
    shutil.copy2(model / "forward.mlir", gpu / "forward.mlir")
    shutil.copy2(harness / "gpu.cpp", gpu / "gpu.cpp")
    run(
        [
            "cmake",
            "-G",
            "Ninja",
            "-S",
            repo / "stack/gvn/gpu",
            "-B",
            gpu / "build",
            f"-DGVN_GPU={gpu}",
        ],
        stdout=subprocess.DEVNULL,
    )
    run(["cmake", "--build", gpu / "build"])
    exposed = json.loads((gpu / "layout.json").read_text())["values"]
    sizes = {"f32": 4, "i8": 1, "i32": 4, "i64": 8}
    run(
        [
            gpu / "gpu",
            model,
            gpu / "values.bin",
            *(f"{len(v['shape'])}:{sizes[v['element']]}:{v['bytes']}" for v in exposed),
        ],
        cwd=gpu,
    )
    offsets, at = {}, 0
    for value in exposed:
        offsets[value["id"]] = (at, value)
        at += value["bytes"]
    if at != (gpu / "values.bin").stat().st_size:
        raise RuntimeError("values.bin does not hold the exposed values")

    def read(id_):
        at, value = offsets[id_]
        with (gpu / "values.bin").open("rb") as file:
            file.seek(at)
            return value, file.read(value["bytes"])

    # ===--- One ELF per slice ---=== #
    # Each slice's values, for this directory's CMakeLists.txt to embed:
    # gvn_in holds them, each 64-byte aligned; gvn_table their offsets and the
    # iteration (patched per instance); gvn_thread the slice's thread, its
    # verify bank and the most bytes one mvin moves on it, from design.pb.
    design = load_design_config(repo, args.design, selection=selection)
    (hart,) = design.device.harts
    for name, layout in layouts.items():
        directory = out / name
        table = [0] * layout["table_words"]
        blob = bytearray()
        for entry in layout["entries"]:
            if entry["kind"] in ("result", "output"):
                if "?" in entry["type"]:
                    raise SystemExit(f"{name}: dynamic {entry['kind']} {entry['type']}")
                continue
            value, data = read(
                entry["id"] if entry["kind"] == "value" else f"arg:{entry['index']}"
            )
            if (
                f"tensor<{'x'.join(map(str, value['shape']))}x{value['element']}>"
                != entry["type"]
            ):
                raise SystemExit(
                    f"{name}: the GPU value {value['id']} is a {value['shape']} "
                    f"{value['element']}, not {entry['type']}"
                )
            at = (len(blob) + 63) // 64 * 64
            blob.extend(bytes(at - len(blob)) + data)
            table[entry["slot"]] = at
        (directory / "gvn_in.bin").write_bytes(bytes(blob) + bytes(64))
        # The unused nonzero sentinel keeps even an all-zero table in file-backed ELF data.
        (directory / "gvn_table.bin").write_bytes(
            struct.pack(f"<{len(table) + 1}q", *table, 1)
        )
        _, expected = read(layout["output"]["id"])
        (directory / "gvn_expect.bin").write_bytes(expected)
        # The readback moves whole 16-byte bank rows.
        output_bytes = (len(expected) + 15) // 16 * 16
        (directory / "gvn_out.bin").write_bytes(bytes([0xA5]) * output_bytes)
        (directory / "gvn_ranges.bin").write_bytes(struct.pack("<QQ", 0, output_bytes))
        thread = design.threads[hart.thread_indices[layout["thread"]]]
        dma_bytes = min(
            thread.mem.bank.entries * 16,
            (thread.transport.prepared.max_ranges - 1) * 4096,
        )
        (directory / "gvn_thread.bin").write_bytes(
            struct.pack(
                "<3Q", layout["thread"], thread.frontend.max_private_vbank, dma_bytes
            )
        )
    elf_build = out / "elf-build"
    run(
        [
            "cmake",
            "-G",
            "Ninja",
            "-S",
            Path(__file__).parent,
            "-B",
            elf_build,
            f"-DDESIGN_CMAKE={config / 'stack/generated/design.cmake'}",
            f"-DMODEL={args.model}",
        ],
        stdout=subprocess.DEVNULL,
    )
    run(
        [
            "cmake",
            "--build",
            elf_build,
            "--parallel",
            str(args.jobs),
            "--target",
            *(out / name / "slice.elf" for name in distinct),
        ]
    )

    # ===--- Each iteration of each slice on BEMU ---=== #
    logs = Path(log_dir(repo, args.design, "gvn", args.model, selection=selection))
    bemu = config / "bin/bebop-bemu"
    bemu_slots = threading.Semaphore(args.jobs)

    def instance(job):
        name, iteration = job
        layout = layouts[name]
        instance_name = f"{name}-iteration{iteration}"
        log = logs / instance_name
        log.mkdir(parents=True, exist_ok=True)
        elf = log / "slice.elf"
        shutil.copy2(out / name / "slice.elf", elf)
        if layout["iteration_slot"] >= 0:
            with elf.open("r+b") as stream:
                stream.seek(table_offset(elf) + layout["iteration_slot"] * 8)
                stream.write(struct.pack("<q", iteration))
        bank = design.threads[
            hart.thread_indices[layout["thread"]]
        ].frontend.max_private_vbank
        size = (out / name / "gvn_expect.bin").stat().st_size
        (log / "readback.json").write_text(
            json.dumps({"bank": bank, "ranges": [[0, size]]}) + "\n"
        )
        bemu_slots.acquire()
        begin = time.monotonic()
        result = subprocess.run(
            [
                bemu,
                "--elf",
                elf,
                "--log-dir",
                log,
                "--tool-profile",
                "--itrace",
                "--mtrace",
            ],
            cwd=log,
            capture_output=True,
            text=True,
            timeout=1200,
        )
        seconds = time.monotonic() - begin
        bemu_slots.release()
        (log / "run.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"BEMU failed: {log}")
        # The whole run (boot, inputs, the slice, the readback): what RTL
        # simulates, the slowest hart's clock.
        simulated = max(
            int(cycles)
            for cycles in re.findall(
                r"^BEMU hart \d+: \d+ instructions, (\d+) cycles$",
                result.stdout + result.stderr,
                re.M,
            )
        )
        report = subprocess.run(
            [
                sys.executable,
                repo / "bbdev/api/steps/gvn/scripts/slice_report.py",
                log,
                str(bank),
                instance_name,
                "reference",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        elapsed = int(report.stdout.split()[-1])
        # The ELF holds the graph arguments the slice reads (the weight
        # packs, up to hundreds of MB); the slice's ELF rebuilds it.
        # slice_report kept the reference: BEMU's output captured in it.
        elf.unlink()
        return {
            "slice": name,
            "instance": instance_name,
            "exit": 0,
            "cycles": elapsed,
            "simulated_cycles": simulated,
            "seconds": round(seconds, 3),
            "iterations": layout["iterations"],
            # For estimates: the slice's ELF, the table slot of its iteration
            # and the times it runs in the whole graph.
            "elf": str(out / name / "slice.elf"),
            "iteration_slot": layout["iteration_slot"],
            # For debug runs: the instance's ELF with BEMU's output captured.
            "reference_elf": str(log / instance_name),
            "repeats": repeats[name],
        }

    jobs = [(name, 0) for name in distinct]
    print(
        f"{len(jobs)} instances: one of each of {len(patterns)} distinct slices of {len(slices)}",
        flush=True,
    )
    begin = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(args.jobs * 2) as pool:
        records = list(pool.map(instance, jobs))
    elapsed = time.monotonic() - begin
    (out / "instances.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    longest = max(records, key=lambda r: r["simulated_cycles"])
    print(
        f"{len(records)} instances of {len(slices)} slices; longest {longest['instance']}: "
        f"{longest['simulated_cycles']} cycles whole run, {longest['cycles']} slice span; "
        f"in {elapsed:.0f} s"
    )
    (logs / "instances.jsonl").write_bytes((out / "instances.jsonl").read_bytes())
    print(f"summary: {out / 'instances.jsonl'}; logs and ELFs: {logs}")


if __name__ == "__main__":
    main()
