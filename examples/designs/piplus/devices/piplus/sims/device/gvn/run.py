"""GVN for the piplus device program: cut the hardware thread subgraphs of
its hart graphs into slices at their DDR values, record with the GPU the
values every launch of the baseline run reads and writes, and run one
instance (a slice, a launch) of each distinct slice as its own NPU program on
BEMU, all in parallel.

  bbdev gvn-bemu --debug '--device piplus
      --model smolvla-base'

Inputs: the compiled model of the device config (`bbdev stack`, the recipe
build: models/<model>/ of the config output with its weights and
observation) and the GPU harness gvn/gpu.cpp of the model recipe.
Outputs under gvn/<model> of the config output; BEMU logs under log/.

NPU side: the recipe's gvn/ tree checks the DDR results of the thread
subgraphs of each hart graph, cuts one slice per check and lowers it with the
recipe's pipeline; its physical bank operations must equal those of the
subgraph in the whole graph at the same place. This directory's
CMakeLists.txt links each instance with readback.cpp and the graphs'
operator images.
GPU side: stack/gvn/program records the values with the model's programs
and graphs.
"""

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from math import prod
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


def wrapped_arguments(path, name):
    """The arguments of the graph `name`_graph as its wrapper `name`, the
    callee of the launches, passes them: for each, ("pack", wrapper argument,
    offset, elements) for a static slice of a wrapper argument, through its
    reshapes, or ("argument", wrapper argument)."""
    text = path.read_text()
    begin = re.search(rf"func\.func @{name}\(", text).start()
    body = text[begin : text.index("\n  }", begin)]
    sources = {}
    for line in body.splitlines():
        line = re.sub(r" loc\([^)]*\)", "", line)
        if match := re.match(
            r"\s*(%\w+) = tensor\.extract_slice %arg(\d+)\[(\d+)\] \[(\d+)\] \[1\]",
            line,
        ):
            sources[match[1]] = ("pack", int(match[2]), int(match[3]), int(match[4]))
        elif match := re.match(r"\s*(%\w+) = tensor\.expand_shape (%\w+) ", line):
            sources[match[1]] = sources[match[2]]
        elif match := re.match(
            rf"\s*%\w+(?::\d+)? = call @{name}_graph\(([^)]*)\)", line
        ):
            return [
                (
                    ("argument", int(operand[len("%arg") :]))
                    if operand.startswith("%arg")
                    else sources[operand]
                )
                for operand in match[1].split(", ")
            ]
    raise SystemExit(f"{name}: no call of {name}_graph")


def scatter(data, dims, container, element):
    """The row-major bytes `data` of shape `dims` placed at the origin of a
    zeroed row-major container of shape `container`."""
    inner = len(dims) - 1
    while inner > 0 and dims[inner] == container[inner]:
        inner -= 1
    chunk = element
    for size in dims[inner:]:
        chunk *= size
    strides = [element] * len(container)
    for dim in range(len(container) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * container[dim + 1]
    out = bytearray(strides[0] * container[0])
    for number, outer in enumerate(
        itertools.product(*(range(size) for size in dims[:inner]))
    ):
        at = sum(i * s for i, s in zip(outer, strides))
        out[at : at + chunk] = data[number * chunk : (number + 1) * chunk]
    return bytes(out)


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
    out.mkdir(parents=True, exist_ok=True)

    gvn.build_plugins()

    # ===--- NPU side: the slices of the thread subgraphs ---=== #
    # The model recipe's tree (its gvn/) checks the thread subgraphs of each
    # hart graph, cuts one slice per check and lowers it.
    npu = out / "npu"
    recipe_build = repo / "stack/models/build" / args.design / selection / args.model
    run(
        ["cmake", "-S", repo / "stack/models", "-B", recipe_build],
        stdout=subprocess.DEVNULL,
    )
    run(["cmake", "--build", recipe_build, "--target", f"{args.model}-gvn-checks"])
    graphs = sorted(
        path.name[: -len("_checks.mlir")] for path in npu.glob("*_checks.mlir")
    )
    for graph in graphs:
        if (npu / f"{graph}_gvn_operators.mlir").read_text() != (
            npu / f"{graph}_model_operators.mlir"
        ).read_text():
            raise SystemExit(
                f"{graph}: slice GP operators differ from the compiled model"
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
            "--target",
            f"{args.model}-gvn",
            "--parallel",
            str(args.jobs),
        ]
    )

    # A check belongs to the function it is in, as the recipe's tree counts.
    slices = []
    for graph in graphs:
        function = None
        for line in (npu / f"{graph}_checks.mlir").read_text().splitlines():
            match = re.match(r"\s*func\.func (?:private )?@(\w+)\(", line)
            if match:
                function, number = match.group(1), 0
            if "gvn.check " in line:
                slices.append((graph, function, number))
                number += 1
    print(f"{len(slices)} slices of {', '.join(graphs)}")
    layouts = {}
    for item in slices:
        graph, function, index = item
        directory = out / f"{function}.{index}"
        operations = function_bank_operations(
            directory / "slice_physical.mlir", function
        )
        # In order, not contiguous: an iteration of a loop loads the operands
        # the whole graph loads once before that loop.
        remaining = iter(
            function_bank_operations(npu / f"{graph}_whole_physical.mlir", function)
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
    sizes = {"f32": 4, "i1": 1, "i8": 1, "i64": 8, "i32": 4}
    # The device programs of the model in order (a pi0-FAST: its prefill,
    # then its decoding step), each with its parameter packs, its leading
    # arguments: files of ddr/ (packs.txt of the harness, as baremetal.cpp
    # passes them).
    packs = {
        name: [model / "ddr" / file for file in files]
        for name, *files in (
            line.split() for line in (harness / "packs.txt").read_text().splitlines()
        )
    }
    programs = list(packs)
    # stack/gvn/program records them, again only when what it reads changed.
    named = json.dumps({g: sorted(v) for g, v in values.items()})
    if not (gpu / "values.json").exists() or (gpu / "values.json").read_text() != named:
        (gpu / "values.json").write_text(named)
    modules = ";".join(
        str(model / f"{name}.mlir")
        for name in (
            *(f"{program}_placed" for program in programs),
            *(f"{graph}_linalg" for graph in graphs),
        )
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
    program = json.loads((gpu / "program.json").read_text())
    numbers = {(v["graph"], v["id"]): v["number"] for v in program["values"]}
    records, steps = {}, {}
    for line in (gpu / "index.txt").read_text().splitlines():
        words = list(map(int, line.split()))
        step, launch, kind, index, element, rank = words[:6]
        records[step, launch, kind, index] = (
            words[6 : 6 + rank],
            words[6 + rank],
            words[7 + rank],
        )
        # An instance of a launch is its first step (a decoding step's the
        # first decoded token).
        steps.setdefault(launch, step)

    # ===--- One ELF per instance, each on BEMU, in parallel ---=== #
    design = load_design_config(repo, args.design, selection=selection)
    logs = Path(log_dir(repo, args.design, "gvn", args.model, selection=selection))
    bemu = config / "bin/bebop-bemu"

    # A public graph is its own entry: its arguments as its wrapper passes
    # them. A private graph's entry is its wrapper, whose arguments are the
    # operands of the launch.
    wrapped = {
        layout["entry"]: wrapped_arguments(
            model / f"{item[0]}_linalg.mlir", layout["graph"][: -len("_graph")]
        )
        for item, layout in layouts.items()
        if layout["entry"] == layout["graph"]
    }

    def prepare(job):
        item, launch, iteration = job
        layout = layouts[item]
        name = f"{item[1]}.{item[2]}-launch{launch}-iteration{iteration}"
        directory = instances / name
        directory.mkdir(parents=True)
        values_file = open(gpu / "values.bin", "rb")

        def read(kind, index):
            dims, offset, size = records[steps[launch], launch, kind, index]
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
                if [d for d in dims if d != 1] == [int(s) for s in shape if s != "1"]:
                    dynamic = [d for d, s in zip(dims, shape) if s == "?"]
                elif "within" in entry:
                    # The full value is a box of the GPU value: that box.
                    within = [
                        int(d)
                        for d in entry["within"][len("tensor<") : -1].split("x")[:-1]
                    ]
                    if [d for d in dims if d != 1] != [d for d in within if d != 1]:
                        raise ValueError(
                            f"output box differs from the GPU value {dims}: {name}: {entry}"
                        )
                    element = sizes[entry["type"][len("tensor<") : -1].split("x")[-1]]
                    box = [int(s) for s in shape]
                    strides = [prod(within[axis + 1 :]) for axis in range(len(within))]
                    expected = b"".join(
                        expected[at * element : (at + box[-1]) * element]
                        for index in itertools.product(
                            *(range(size) for size in box[:-1])
                        )
                        for at in [
                            sum(
                                (o + i) * st
                                for o, i, st in zip(
                                    entry["offsets"], (*index, 0), strides
                                )
                            )
                        ]
                    )
                    dynamic = []
                    output_bytes = len(expected)
                    ranges = [(0, output_bytes)]
                else:
                    rows = prod(dims)
                    if (
                        len(shape) != 2
                        or shape[1] != "4"
                        or shape[0] not in ("?", str(rows))
                    ):
                        raise ValueError(
                            f"output type differs from the GPU value {dims}: {name}: {entry}"
                        )
                    element = sizes[entry["type"][len("tensor<") : -1].split("x")[-1]]
                    dynamic = [rows] if shape[0] == "?" else []
                    output_bytes = rows * 4 * element
                    ranges = [(row * 4 * element, element) for row in range(rows)]
                table[entry["slot"] : entry["slot"] + len(dynamic)] = dynamic
                continue
            if entry["kind"] == "argument":
                # The graph's argument as the wrapper passes it: a slice of a
                # pack of the program, or an operand of the launch.
                source = (
                    wrapped[layout["entry"]][entry["index"]]
                    if layout["entry"] in wrapped
                    else ("argument", entry["index"])
                )
                operand = operands[source[1]]
                if "pack" in operand:
                    element = sizes[entry["element"]]
                    offset = operand["offset"]
                    elements = operand["elements"]
                    if source[0] == "pack":
                        offset += source[2]
                        elements = source[3]
                    with open(
                        packs[program["launches"][launch]["program"]][operand["pack"]],
                        "rb",
                    ) as pack:
                        pack.seek(offset * element)
                        data = pack.read(elements * element)
                    dims = [elements]
                    at = place(data, offset * element % 64)
                else:
                    if source[0] != "argument":
                        raise SystemExit(f"{name}: a pack slice of a computed operand")
                    dims, data = read(0, source[1])
                    if "container" in entry:
                        # The argument is the leading view of a container
                        # (its strides): its rows are placed in it.
                        container = [
                            int(d)
                            for d in entry["container"][len("tensor<") : -1].split("x")[
                                :-1
                            ]
                        ]
                        data = scatter(data, dims, container, sizes[entry["element"]])
                    at = place(data)
            else:
                # Unit dimensions aside, the same row-major bytes.
                dims, data = read(
                    1, numbers[entry.get("graph", layout["graph"]), entry["id"]]
                )
                if [d for d in dims if d != 1] != [int(s) for s in shape if s != "1"]:
                    raise ValueError(
                        f"fixture type differs from the slice input: {name}: {entry}"
                    )
                at = place(data)
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
        bank, thread = struct.unpack(
            "<4Q", (directory / "gvn_thread.bin").read_bytes()
        )[2:0:-1]
        ranges = json.loads((directory / "readback.json").read_text())
        log = logs / name
        log.mkdir(parents=True, exist_ok=True)
        shutil.copy2(directory / "slice.elf", log / "slice.elf")
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
        (log / "slice.elf").unlink()
        return {
            "instance": name,
            "exit": 0,
            "cycles": elapsed,
            "simulated_cycles": simulated,
            "seconds": round(seconds, 3),
        }

    bemu_slots = threading.Semaphore(args.jobs)
    # Debug runs seek coverage: one instance of each distinct slice (its
    # first launch and iteration), slices compiled to the same object being
    # one.
    # A graph the programs never launch (a partition's join no layer needs)
    # has no instance.
    first = {}
    for launch in program["launches"]:
        first.setdefault(launch["callee"] + "_graph", launch["index"])
    unlaunched = sorted({layouts[item]["graph"] for item in slices} - set(first))
    # The repeats of a distinct slice: the launches of its graph times its
    # iterations, over every slice compiled to it.
    launches = {}
    for launch in program["launches"]:
        graph = launch["callee"] + "_graph"
        launches[graph] = launches.get(graph, 0) + 1
    # Slices of thread graphs with the same body (the same computation on
    # another thread, other rows or another part of a parallel group: the
    # hart's slice code differs only in those) are one.
    bodies = {
        function: hashlib.sha256(
            re.sub(
                r" loc\((?:[^()]|\([^()]*\))*\)|hart\.parallel_part = \d+ : i64",
                "",
                body,
            ).encode()
        ).hexdigest()
        for graph in graphs
        for function, body in re.findall(
            r"^  func\.func private @(\w+)\((.*?)^  \}",
            (npu / f"{graph}_hart.mlir").read_text(),
            re.M | re.S,
        )
    }
    patterns, repeats = {}, {}
    for item in slices:
        if layouts[item]["graph"] not in first:
            continue
        # The part of the thread graph the slice keeps.
        code = (bodies[item[1]], item[2])
        patterns.setdefault(code, item)
        repeats[patterns[code]] = (
            repeats.get(patterns[code], 0)
            + launches[layouts[item]["graph"]] * layouts[item]["iterations"]
        )
    jobs = [(item, first[layouts[item]["graph"]], 0) for item in patterns.values()]
    print(
        f"{len(jobs)} instances: one of each of {len(patterns)} distinct slices of {len(slices)}"
        f"{f'; never launched: {unlaunched}' if unlaunched else ''}",
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

    # The instance inputs on disk at once, a bound for the shared /home.
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
    elapsed = time.monotonic() - begin
    records = [
        {
            "slice": f"{job[0][1]}.{job[0][2]}",
            **results[f"{job[0][1]}.{job[0][2]}-launch{job[1]}-iteration{job[2]}"],
            # For estimates: the instance's ELF (its iteration in it) and the
            # times its slice runs in the whole program.
            "elf": str(
                logs
                / f"{job[0][1]}.{job[0][2]}-launch{job[1]}-iteration{job[2]}"
                / f"{job[0][1]}.{job[0][2]}-launch{job[1]}-iteration{job[2]}"
            ),
            # For debug runs: the same ELF, BEMU's output captured in it.
            "reference_elf": str(
                logs
                / f"{job[0][1]}.{job[0][2]}-launch{job[1]}-iteration{job[2]}"
                / f"{job[0][1]}.{job[0][2]}-launch{job[1]}-iteration{job[2]}"
            ),
            "iteration_slot": -1,
            "repeats": repeats[job[0]],
        }
        for job in jobs
    ]
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
