"""GVN for a device program (poly): cut every hardware thread of the hart
graphs into slices at its DDR values, record with the GPU the values every
launch of every step of the baseline run reads and writes, and run one
instance (a slice, a step, a launch) of each distinct slice as its own NPU
program on BEMU, all in parallel.

  bbdev gvn-bemu --debug '--device poly
      --model qwen3-0-6b'

Inputs: the compiled model of the device config (`bbdev stack`, the recipe
build: models/<model>/ of the config output with its prompt) and the
harness gvn/gpu.cpp of the model recipe. Outputs under
gvn/<model> of the config output; BEMU logs under log/.

NPU side: the recipe's gvn/ tree checks the DDR results of the thread
subgraphs, cuts one slice per check and lowers it with the recipe's
pipeline; its physical bank operations must equal those of the subgraph in
the whole graph at the same place. This directory's CMakeLists.txt links
each instance with readback.cpp.
GPU side: stack/gvn/program records the values with the model's programs
and graphs.
"""

import argparse
import concurrent.futures
import hashlib
import json
from itertools import zip_longest
from math import prod
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[8] / "bbdev/api/steps/gvn/scripts")
)
import gvn  # noqa: E402
from utils.path import load_design_config, log_dir  # noqa: E402

repo = gvn.repo
run = gvn.run


def function_bank_operations(path, name):
    """The physical bank operations of function `name`."""
    text = path.read_text()
    begin = re.search(rf"func\.func (?:private )?@{name}\(", text).start()
    following = re.search(r"\n  func\.func ", text[begin + 1 :])
    body = text[begin : begin + 1 + following.start()] if following else text[begin:]
    return gvn.bank_operations(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--model", required=True)
    # BEMU processes at once.
    parser.add_argument("--jobs", type=int, default=96)
    args = parser.parse_args()
    args.design = Path(args.config).parts[0]
    stem = Path(args.config).stem

    config = (
        repo
        / "examples/designs"
        / args.design
        / "output/sims/device"
        / args.device
        / stem
    )
    model = config / "models" / args.model
    prompt = model / "ddr"
    harness = repo / "examples/models" / args.design / args.model / "gvn"
    out = config / "gvn" / args.model
    out.mkdir(parents=True, exist_ok=True)

    gvn.build_plugins()

    # ===--- NPU side: the slices of the thread subgraphs ---=== #
    # The model recipe's tree (its gvn/) checks the thread subgraphs of the
    # hart graph, cuts one slice per check and lowers it.
    npu = out / "npu"
    recipe_build = (
        repo
        / "stack/models/build"
        / args.design
        / f"device/{args.device}/{stem}"
        / args.model
    )
    run(
        ["cmake", "-S", repo / "stack/models", "-B", recipe_build],
        stdout=subprocess.DEVNULL,
    )
    run(["cmake", "--build", recipe_build, "--target", f"{args.model}-gvn-checks"])
    if (npu / "gvn_operators.mlir").read_text() != (
        npu / "model_operators.mlir"
    ).read_text():
        raise SystemExit("slice GP operators differ from the compiled model")
    run(
        ["cmake", "-S", repo / "stack/models", "-B", recipe_build],
        stdout=subprocess.DEVNULL,
    )
    run(
        [
            "cmake",
            "--build",
            recipe_build,
            "--target",
            f"{args.model}-gvn",
            "--parallel",
            str(args.jobs),
        ]
    )

    # A check belongs to the function it is in, as the recipe's tree counts.
    slices = []
    function = None
    for line in (npu / "checks.mlir").read_text().splitlines():
        match = re.match(r"\s*func\.func (?:private )?@(\w+)\(", line)
        if match:
            function, number = match.group(1), 0
        if "gvn.check " in line:
            slices.append((function, number))
            number += 1
    print(f"{len(slices)} slices")
    layouts = {}
    for item in slices:
        function, index = item
        directory = out / f"{function}.{index}"
        operations = function_bank_operations(
            directory / "slice_physical.mlir", function
        )
        # In order, not contiguous: an iteration of a loop loads the operands
        # the whole graph loads once before that loop.
        remaining = iter(
            function_bank_operations(npu / "whole_physical.mlir", function)
        )
        if not all(operation in remaining for operation in operations):
            raise SystemExit(
                f"{function}#{index}: its physical bank operations differ from those of the whole graph"
            )
        layouts[item] = json.loads((directory / "layout.json").read_text())
    print("physical banks: every slice appears in its subgraph in the whole graph")

    # ===--- GPU side: the values of every launch of every step ---=== #
    gpu = out / "gpu"
    gpu.mkdir(exist_ok=True)
    values = {}
    for layout in layouts.values():
        for entry in layout["entries"]:
            if entry["kind"] == "value":
                values.setdefault(entry.get("graph", layout["graph"]), set()).add(
                    entry["id"]
                )
        values.setdefault(layout["graph"], set()).add(layout["output"]["id"])
    # stack/gvn/program records them, again only when what it reads changed.
    named = json.dumps({g: sorted(v) for g, v in values.items()})
    if not (gpu / "values.json").exists() or (gpu / "values.json").read_text() != named:
        (gpu / "values.json").write_text(named)
    modules = ";".join(
        str(model / f"{name}.mlir")
        for name in ("program_placed", "poly_linalg", "reduce_linalg")
    )
    run(
        [
            "cmake",
            "-G",
            "Ninja",
            "-S",
            repo / "stack/gvn/program",
            "-B",
            gpu / "build",
            f"-DDESIGN_CMAKE={config / 'stack/generated/design.cmake'}",
            f"-DGVN_GPU={gpu}",
            f"-DMODULES={modules}",
            f"-DMODEL={model}",
            f"-DHARNESS={harness}",
        ],
        stdout=subprocess.DEVNULL,
    )
    run(["cmake", "--build", gpu / "build"])
    # The parameter packs, the program's first arguments: as baremetal.cpp
    # passes them, the F32, INT8 and INT32 packs of the prompt.
    pack_count = int((gpu / "program.txt").read_text().split()[2])
    packs = [
        prompt / name
        for name in ("pack-f32.data", "pack-i8.data", "pack-i32.data")[:pack_count]
    ]
    sizes = {"f32": 4, "i1": 1, "i8": 1, "i64": 8, "i32": 4}
    program = json.loads((gpu / "program.json").read_text())
    numbers = {(v["graph"], v["id"]): v["number"] for v in program["values"]}
    records = {}
    for line in (gpu / "index.txt").read_text().splitlines():
        words = list(map(int, line.split()))
        step, launch, kind, index, element, rank = words[:6]
        records[step, launch, kind, index] = (
            words[6 : 6 + rank],
            words[6 + rank],
            words[7 + rank],
        )
    # The steps the GPU ran: the recipe's.
    steps = 1 + max(key[0] for key in records)

    # ===--- One ELF per instance, each on BEMU, in parallel ---=== #
    # The iterations of a slice in a launch: a slice over the rows of its
    # output (iterations -1) has a batch of them each, as many as the GPU
    # value of that launch has.
    def iterations(item, step, launch):
        layout = layouts[item]
        if layout["iterations"] >= 0:
            return layout["iterations"]
        dims = records[
            step, launch, 1, numbers[layout["graph"], layout["output"]["id"]]
        ][0]
        return -(-dims[layout["rows"]] // layout["iteration_batch"])

    def instances_of(items):
        instances = [
            (item, step, launch["index"], iteration)
            for item in items
            for launch in program["launches"]
            if launch["callee"] == layouts[item]["entry"]
            for step in range(steps)
            for iteration in range(iterations(item, step, launch["index"]))
        ]
        # Cover every slice and step before repeating layers and loop iterations.
        groups = {}
        for instance_ in instances:
            groups.setdefault((instance_[0], instance_[1]), []).append(instance_)
        for group in groups.values():
            group.sort(key=lambda instance_: (instance_[3], instance_[2]))
        return [
            instance_
            for batch_ in zip_longest(*groups.values())
            for instance_ in batch_
            if instance_ is not None
        ]

    design = load_design_config(
        repo, args.design, selection=f"device/{args.device}/{stem}"
    )
    logs = Path(
        log_dir(
            repo,
            args.design,
            "gvn",
            args.model,
            selection=f"device/{args.device}/{stem}",
        )
    )
    bemu = config / "bin/bebop-bemu"

    def prepare(job):
        item, step, launch, iteration = job
        layout = layouts[item]
        name = f"{item[0]}.{item[1]}-step{step}-launch{launch}-iteration{iteration}"
        directory = instances / name
        directory.mkdir(parents=True)
        values_file = open(gpu / "values.bin", "rb")

        def read(kind, index):
            dims, offset, size = records[step, launch, kind, index]
            values_file.seek(offset)
            return dims, values_file.read(size)

        operands = {o["index"]: o for o in program["launches"][launch]["operands"]}
        table = [0] * layout["table_words"]
        blob = bytearray()

        def place(data, alignment=0):
            at = (len(blob) + 63) // 64 * 64 + alignment
            blob.extend(bytes(at - len(blob)) + data)
            return at

        # The slice writes the GPU value whole, or (an output entry) its
        # partition of it.
        _, expected = read(1, numbers[layout["graph"], layout["output"]["id"]])
        output_bytes, ranges = len(expected), [(0, len(expected))]
        for entry in layout["entries"]:
            shape = entry["type"][len("tensor<") : -1].split("x")[:-1]
            if entry["kind"] == "result":
                dims, _ = read(2, entry["index"])
                dynamic = [d for d, s in zip(dims, shape) if s == "?"]
                table[entry["slot"] : entry["slot"] + len(dynamic)] = dynamic
                continue
            if entry["kind"] == "output":
                # The full value the thread's partition is placed in: gvn_out
                # has the sizes of the GPU value it is compared with, but for a
                # grouped reduction, whose threads fill rows of 4 with the
                # value, flattened, in column 0: those rows, each read back at
                # its column 0.
                dims, _ = read(1, numbers[layout["graph"], layout["output"]["id"]])
                if len(dims) == len(shape) and all(
                    s == "?" or int(s) == d for d, s in zip(dims, shape)
                ):
                    dynamic = [d for d, s in zip(dims, shape) if s == "?"]
                else:
                    rows = prod(dims)
                    if shape != ["?", "4"]:
                        raise ValueError(
                            f"output type differs from the GPU value {dims}: {name}: {entry}"
                        )
                    element = sizes[entry["type"][len("tensor<") : -1].split("x")[-1]]
                    dynamic = [rows]
                    output_bytes = rows * 4 * element
                    ranges = [(row * 4 * element, element) for row in range(rows)]
                table[entry["slot"] : entry["slot"] + len(dynamic)] = dynamic
                continue
            if entry["kind"] == "argument" and "pack" in operands[entry["index"]]:
                operand = operands[entry["index"]]
                element = sizes[entry["element"]]
                with open(packs[operand["pack"]], "rb") as pack:
                    pack.seek(operand["offset"] * element)
                    data = pack.read(operand["elements"] * element)
                dims = [operand["elements"]]
                at = place(data, operand["offset"] * element % 64)
            elif entry["kind"] == "argument":
                dims, data = read(0, entry["index"])
                at = place(data)
            else:
                dims, data = read(
                    1, numbers[entry.get("graph", layout["graph"]), entry["id"]]
                )
                at = place(data)
            if (
                len(dims) != len(shape)
                or any(s != "?" and int(s) != d for d, s in zip(dims, shape))
                or len(data) != prod(dims) * sizes[entry["element"]]
            ):
                raise ValueError(
                    f"fixture type differs from the slice input: {name}: {entry}"
                )
            dynamic = [d for d, s in zip(dims, shape) if s == "?"]
            table[entry["slot"]] = at
            table[entry["slot"] + 1 : entry["slot"] + 1 + len(dynamic)] = dynamic
        if layout["iteration_slot"] >= 0:
            table[layout["iteration_slot"]] = iteration
        (directory / "gvn_in.bin").write_bytes(bytes(blob) + bytes(64))
        (directory / "gvn_table.bin").write_bytes(
            struct.pack(f"<{len(table)}q", *table)
        )
        (directory / "gvn_expect.bin").write_bytes(expected)
        # The readback moves whole 16-byte bank rows.
        (directory / "gvn_out.bin").write_bytes(
            bytes([0xA5]) * ((output_bytes + 15) // 16 * 16)
        )
        (directory / "gvn_ranges.bin").write_bytes(
            b"".join(struct.pack("<QQ", *r) for r in ranges)
        )
        (directory / "readback.json").write_text(
            json.dumps([[first, first + length] for first, length in ranges])
        )
        hart = program["launches"][launch]["hart"]
        thread = design.threads[
            design.device.harts[hart].thread_indices[layout["thread"]]
        ]
        dma_bytes = min(
            thread.mem.bank.entries * 16,
            (thread.transport.prepared.max_ranges - 1) * 4096,
        )
        (directory / "gvn_thread.bin").write_bytes(
            struct.pack(
                "<4Q",
                hart,
                layout["thread"],
                thread.frontend.max_private_vbank,
                dma_bytes,
            )
        )
        values_file.close()
        # The inputs are held three times (the input file, the object, the
        # ELF) until the instance has run.
        return name, 3 * len(blob)

    def instance(name):
        directory = instances / name
        elf = directory / "slice.elf"
        bank, thread = struct.unpack(
            "<4Q", (directory / "gvn_thread.bin").read_bytes()
        )[2:0:-1]
        ranges = json.loads((directory / "readback.json").read_text())
        log = logs / name
        log.mkdir(parents=True, exist_ok=True)
        # A hard link, not a copy: an instance holds up to a few hundred MB.
        os.link(elf, log / "slice.elf")
        (log / "readback.json").write_text(
            json.dumps({"bank": bank, "ranges": ranges, "thread": thread}) + "\n"
        )
        shutil.rmtree(directory)
        bemu_slots.acquire()
        begin = time.monotonic()
        result = subprocess.run(
            [
                bemu,
                "--elf",
                log / "slice.elf",
                "--log-dir",
                log,
                "--itrace",
                "--mtrace",
                "--tool-profile",
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
        # The report in a process of its own: the traces parse in parallel.
        report = subprocess.run(
            [
                sys.executable,
                repo / "bbdev/api/steps/gvn/scripts/slice_report.py",
                log,
                str(bank),
                name,
                "reference",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        elapsed = int(report.stdout.split()[-1])
        return name, 0, ["BEMU reference captured"], elapsed, simulated, seconds

    bemu_slots = threading.Semaphore(args.jobs)
    # Debug runs seek coverage: one instance of each distinct slice (its
    # first launch, step and iteration). Slices of thread graphs with the
    # same body (the same computation on another thread, other rows or
    # another part of a parallel group: the hart's slice code differs only in
    # those) are one.
    # The repeats of a distinct slice: the instances (launches, steps and
    # iterations) of every slice that is it.
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
    patterns, repeats = {}, {}
    for item in slices:
        # The part of the thread graph the slice keeps.
        code = (bodies[item[0]], item[1])
        patterns.setdefault(code, item)
        repeats[patterns[code]] = repeats.get(patterns[code], 0) + len(
            instances_of([item])
        )
    jobs = [instances_of([item])[0] for item in patterns.values()]
    print(
        f"{len(jobs)} instances: one of each of {len(patterns)} distinct slices of {len(slices)}",
        flush=True,
    )
    instances = out / "instances"
    shutil.rmtree(instances, ignore_errors=True)
    instances.mkdir()
    elf_build = out / "elf-build"
    results = {}
    begin = time.monotonic()

    def flush(batch):
        # This directory's CMakeLists.txt links the batch's ELFs.
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
            ["cmake", "--build", elf_build, "--parallel", str(args.jobs)],
            stdout=subprocess.DEVNULL,
        )
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
            results.update(zip(batch, pool.map(instance, batch)))

    # The instance inputs on disk at once: an LM head instance holds ~240 MB,
    # and 16 of them in flight (a 4 GB bound) took 45 GB of the shared /home
    # within 20 s, freed when they ended.
    batch, held = [], 0
    for job in jobs:
        name, size = prepare(job)
        if batch and held + size > 1 << 30:
            flush(batch)
            batch, held = [], 0
        batch.append(name)
        held += size
    if batch:
        flush(batch)
    records = []
    for job in jobs:
        name = f"{job[0][0]}.{job[0][1]}-step{job[1]}-launch{job[2]}-iteration{job[3]}"
        _, code, report, cycles, simulated, seconds = results[name]
        records.append(
            {
                "slice": f"{job[0][0]}.{job[0][1]}",
                "instance": name,
                "exit": code,
                "cycles": cycles,
                "simulated_cycles": simulated,
                "seconds": round(seconds, 3),
                "report": report,
                # For estimates: the instance's ELF (its iteration in it) and
                # the times its slice runs in the whole program.
                "elf": str(logs / name / name),
                # For debug runs: the same ELF, BEMU's output captured in it.
                "reference_elf": str(logs / name / name),
                "iteration_slot": -1,
                "repeats": repeats[job[0]],
            }
        )
    elapsed = time.monotonic() - begin
    (out / "instances.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    longest = max(records, key=lambda r: r["simulated_cycles"])
    print(
        f"{len(records)} instances; longest {longest['instance']}: {longest['simulated_cycles']} cycles "
        f"whole run, {longest['cycles']} slice span; in {elapsed:.0f} s"
    )
    (logs / "instances.jsonl").write_bytes((out / "instances.jsonl").read_bytes())
    print(f"summary: {out / 'instances.jsonl'}; logs and ELFs: {logs}")


if __name__ == "__main__":
    main()
