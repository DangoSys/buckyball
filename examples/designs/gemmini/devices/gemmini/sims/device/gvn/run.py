"""GVN for Gemmini: cut a model compiled for a device config into slices, one
per operation Gemmini takes (a matmul or a convolution, from DDR), and build
one NPU program (ELF) per slice with GPU input fixtures, then capture each
slice's BEMU output as the reference for Verilator.

  bbdev gvn-bemu --debug '--device gemmini --model lenet'

Inputs: the compiled model of the device config (the recipe build:
models/<model>/ of the config output) and the harnesses in gvn/ of the model
recipe. Outputs under gvn/<model> of the config output; BEMU logs under log/.

NPU side: gvn-insert-op-checks marks the result of every operation of the
linalg graph Gemmini takes, gvn-slice-op keeps one of them alone, reading
its operands from gvn_in; the slice is lowered with the recipe's pipeline.
GPU side: gvn-expose makes the linalg graph return the operands and results
of those operations, compiled with the nv pipeline and run on the GPU.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[8] / "bbdev/api/steps/gvn/scripts")
)
from gvn import build_plugins, repo, run  # noqa: E402

from utils.path import log_dir  # noqa: E402

# AlexNet's 10M-cycle BEMU matmul tiles exceeded 30 minutes on RTL.
budget = 1_000_000
# The operations a slice of which is cut into tiles when longer.
products = ("linalg.matmul", "linalg.batch_matmul", "linalg.batch_matmul_transpose_b")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
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
    recipe = repo / "stack/models/archs/gemmini" / args.model
    model = recipe / "output"
    out = config / "gvn" / args.model
    out.mkdir(parents=True, exist_ok=True)

    build_plugins()
    npu = out / "npu"

    # ===--- NPU side: checks and slices ---=== #
    # The recipe's tree (gvn/ of the recipe) checks the linalg graph, then,
    # configured again, cuts one slice per check, each the tiles of its
    # operation slice<i>/parts names, and lowers it.
    recipe_build = (
        repo
        / "stack/models/build"
        / args.design
        / f"device/{args.device}/{stem}"
        / args.model
    )

    def configure():
        # A slice starts whole; an earlier run's tiles stand.
        if (npu / "checks.mlir").exists():
            checks = re.findall(r"gvn\.check ", (npu / "checks.mlir").read_text())
            for index in range(len(checks)):
                (out / f"slice{index}").mkdir(exist_ok=True)
                parts = out / f"slice{index}" / "parts"
                if not parts.exists():
                    parts.write_text("1\n")
        run(
            ["cmake", "-S", repo / "stack/models", "-B", recipe_build],
            stdout=subprocess.DEVNULL,
        )

    configure()
    run(["cmake", "--build", recipe_build, "--target", f"{args.model}-gvn-checks"])
    count = len(re.findall(r"gvn\.check ", (npu / "checks.mlir").read_text()))

    def build(target):
        configure()
        run(
            [
                "cmake",
                "--build",
                recipe_build,
                "--target",
                f"{args.model}-{target}",
                "--parallel",
                str(args.jobs),
            ]
        )

    def layouts():
        return [
            json.loads((out / f"slice{index}" / "layout.json").read_text())
            for index in range(count)
        ]

    build("gvn")
    slices = layouts()
    print(f"{count} slices: {', '.join(s['operation'] for s in slices)}")

    # ===--- GPU side: the checked values ---=== #
    # The operands and the result of every slice, computed by stack/gvn/gpu
    # with the entry and the model's host harness.
    ids = sorted(
        {
            *(layout["output"]["id"] for layout in slices),
            *(entry["id"] for layout in slices for entry in layout["entries"]),
        }
    )
    gpu = out / "gpu"
    gpu.mkdir(exist_ok=True)
    (gpu / "ids").write_text(",".join(ids))
    shutil.copy2(model / "subgraph0_linalg.mlir", gpu / "linalg.mlir")
    shutil.copy2(model / "forward.mlir", gpu / "forward.mlir")
    shutil.copy2(recipe / "gvn/gpu.cpp", gpu / "gpu.cpp")
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
    sizes = {"f32": 4, "f16": 2, "i8": 1, "i32": 4, "i64": 8}
    run(
        [
            gpu / "gpu",
            model,
            gpu / "values.bin",
            *(
                f"{len(value['shape'])}:{sizes[value['element']]}:{value['bytes']}"
                for value in exposed
            ),
        ],
        cwd=gpu,
    )
    blob = (gpu / "values.bin").read_bytes()
    values = {}
    for value in exposed:
        values[value["id"]] = blob[: value["bytes"]]
        blob = blob[value["bytes"] :]
    shapes = {value["id"]: value for value in exposed}

    def cuts(layout):
        """The loops a slice's tiles cut, in order, as gvn-slice-op cuts them:
        each one's dimension of the value, extent (a dynamic one's from the
        GPU value) and tile size."""
        shape = shapes[layout["output"]["id"]]["shape"]
        left, result = layout["tiles"]["parts"], []
        for loop in layout["tiles"]["loops"]:
            extent = loop["extent"] if loop["extent"] >= 0 else shape[loop["dim"]]
            cut = min(left, extent)
            result.append((loop["dim"], extent, -(-extent // cut)))
            left = -(-left // cut)
        return result

    # ===--- One ELF per slice ---=== #
    # The operands of each slice, for the recipe's tree to embed: gvn_in
    # holds them, each 64-byte aligned; gvn_table their offsets; gvn_expect
    # the slice's result: of a tiled one, the first tile of the value along
    # the dimension cut.
    def fixtures(layout):
        index = layout["index"]
        directory = out / f"slice{index}"
        inputs = bytearray()
        table = [0] * layout["table_words"]
        for entry in layout["entries"]:
            given = shapes[entry["id"]]
            if (
                f"tensor<{'x'.join(map(str, given['shape']))}x{given['element']}>"
                != entry["type"]
            ):
                raise SystemExit(
                    f"slice {index}: the GPU value {entry['id']} is a "
                    f"{given['shape']} {given['element']}, not {entry['type']}"
                )
            at = (len(inputs) + 63) // 64 * 64
            inputs.extend(bytes(at - len(inputs)) + values[entry["id"]])
            table[entry["slot"]] = at
        (directory / "gvn_in.bin").write_bytes(bytes(inputs) + bytes(64))
        # The unused nonzero sentinel keeps even an all-zero table in file-backed ELF data.
        (directory / "gvn_table.bin").write_bytes(
            struct.pack(f"<{len(table) + 1}q", *table, 1)
        )
        output = layout["output"]["id"]
        expected = values[output]
        tiles = layout["tiles"]
        if tiles["parts"] > 1:
            # The first tile: the leading size of each dimension cut.
            shape = shapes[output]["shape"]
            kept = list(shape)
            for dim, _, size in cuts(layout):
                kept[dim] = size

            def first(data, depth):
                if kept[depth:] == shape[depth:]:
                    return data
                stride = len(data) // shape[depth]
                return b"".join(
                    first(data[row * stride : (row + 1) * stride], depth + 1)
                    for row in range(kept[depth])
                )

            expected = first(expected, 0)
        (directory / "gvn_expect.bin").write_bytes(expected)

    for layout in slices:
        fixtures(layout)
    build("gvn-elf")

    # Debug runs seek coverage: one instance of each distinct slice, slices
    # of the same operation on operands of the same types (the same slice
    # program) being one. Each runs once in the whole graph: the repeats of
    # a distinct slice are the slices that are it, times the tiles of its
    # operation.
    def distinct():
        patterns, repeats, members = {}, {}, {}
        for layout in slices:
            code = hashlib.sha256(
                (out / f"slice{layout['index']}" / "slice.mlir").read_bytes()
            ).hexdigest()
            patterns.setdefault(code, layout["index"])
            members.setdefault(patterns[code], []).append(layout["index"])
            tiles = layout["tiles"]
            repeats[patterns[code]] = repeats.get(patterns[code], 0) + (
                math.prod(extent / size for _, extent, size in cuts(layout))
                if tiles["parts"] > 1
                else 1
            )
        return patterns, repeats, members

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

    def instance(index):
        name = f"{args.design}-{args.model}-gvn-slice{index}"
        log = logs / name
        shutil.rmtree(log, ignore_errors=True)
        log.mkdir(parents=True)
        elf = log / name
        shutil.copy2(out / f"slice{index}" / "slice.elf", elf)
        begin = time.monotonic()
        result = subprocess.run(
            [bemu, "--elf", elf, "--log-dir", log],
            cwd=log,
            capture_output=True,
            text=True,
        )
        seconds = time.monotonic() - begin
        (log / "run.log").write_text(result.stdout + result.stderr)
        if result.returncode == 0:
            (log / "reference.elf").rename(elf)
        report = [
            line
            for line in result.stdout.splitlines()
            if line.startswith("Memory reference captured")
        ]
        if result.returncode == 0 and len(report) != 1:
            raise RuntimeError(
                f"{name}: expected one BEMU reference capture, got {report}"
            )
        # The slice's own span, trace id INT64_MAX.
        (timing,) = log.glob(
            "trace/controller-*/thread-*/cycle/trace-9223372036854775807.txt"
        )
        cycles = re.findall(r"^elapsed (\d+)$", timing.read_text(), re.MULTILINE)
        return (
            name,
            result.returncode,
            report,
            int(cycles[0]) if cycles else -1,
            seconds,
            index,
        )

    # A slice longer on BEMU than one Verilator simulation of this design
    # should take is cut into more tiles, until none is.
    begin = time.monotonic()
    patterns, repeats, members = distinct()
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        runs = {run_[5]: run_ for run_ in pool.map(instance, patterns.values())}

    # Products are cut, convolutions are not: Gemmini takes square images
    # only, a tile of rows is no longer one. A slice of one iteration per
    # tile is cut no further.
    # The most tiles of a slice: one iteration of each loop cut.
    def limit(layout):
        return math.prod(extent for _, extent, _ in cuts(layout))

    def cut(index):
        tiles = slices[index]["tiles"]
        return slices[index]["operation"] in products and (
            tiles["parts"] < limit(slices[index])
        )

    while long := [run_ for run_ in runs.values() if run_[3] > budget and cut(run_[5])]:
        changed = set()
        for name, _, _, cycles, _, index in long:
            parts = out / f"slice{index}" / "parts"
            tiles = int(parts.read_text()) * -(-cycles // budget)
            if slices[index]["tiles"]["parts"] > 1:
                tiles = min(tiles, limit(slices[index]))
            print(f"{name}: {cycles} cycles, cut into {tiles} tiles")
            for member in members[index]:
                (out / f"slice{member}" / "parts").write_text(f"{tiles}\n")
                changed.add(member)
        build("gvn")
        slices = layouts()
        for index in changed:
            fixtures(slices[index])
        build("gvn-elf")
        patterns, repeats, members = distinct()
        runs = {i: runs[i] for i in patterns.values() if i in runs and i not in changed}
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
            runs.update(
                {
                    r[5]: r
                    for r in pool.map(
                        instance, [i for i in patterns.values() if i not in runs]
                    )
                }
            )
    elapsed = time.monotonic() - begin
    runs = [runs[index] for index in patterns.values()]
    (logs / "slices.json").write_text(json.dumps(slices, indent=2) + "\n")
    manifest = "".join(
        json.dumps(
            {
                "instance": name,
                "exit": code,
                "cycles": cycles,
                "seconds": round(seconds, 3),
                "report": report,
                # For estimates: the slice's ELF, the table slot of its
                # iteration (none) and the times it runs in the whole graph.
                "elf": str(out / f"slice{index}" / "slice.elf"),
                "reference_elf": str(logs / name / name),
                "iteration_slot": -1,
                "repeats": repeats[index],
            }
        )
        + "\n"
        for name, code, report, cycles, seconds, index in runs
    )
    (logs / "instances.jsonl").write_text(manifest)
    published = out / f"instances-{logs.name}.jsonl.tmp"
    published.write_text(manifest)
    published.replace(out / "instances.jsonl")
    for name, code, report, cycles, seconds, _ in runs:
        if code or cycles < 0:
            print(f"{name}: exit {code}, {seconds:.2f} s")
            for line in report:
                print(f"  {line}")
    failed = [run_ for run_ in runs if run_[1] or run_[3] < 0]
    longest = max(runs, key=lambda run_: run_[3])
    print(
        f"{len(runs)} instances, {len(failed)} failed; {sum(run_[3] for run_ in runs)} "
        f"cycles in all; longest {longest[0]}: {longest[3]} cycles, {longest[4]:.2f} s; "
        f"all in parallel: {elapsed:.2f} s"
    )
    print(f"summary: {out / 'instances.jsonl'}; logs and ELFs: {logs}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
