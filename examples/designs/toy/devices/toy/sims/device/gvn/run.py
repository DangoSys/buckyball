"""GVN for Toy: cut a model compiled for a device config into slices and
build one self-contained NPU program (ELF) per slice with GPU input fixtures,
then capture each slice's BEMU output as the reference for Verilator.

  bbdev gvn-bemu --debug '--device toy --model lenet'

Inputs: the compiled model of the device config (`bbdev stack` and the
recipe build: models/<model>/ of the config output, the hart compiler
plugin) and the harnesses in gvn/ of the model recipe. Outputs under
gvn/<model> of the config output; BEMU logs under log/.

NPU side: the linalg graph is compiled by the hart's convert-linalg-to-acc
with its locations, gvn-insert-checks marks every DDR result of a hart
computation, gvn-slice keeps one span between two checks. Each slice is
lowered with the recipe's pipeline; its physical bank operations must equal
those of the whole graph at the same place.
GPU side: gvn-expose makes the linalg graph return the checked values,
compiled with the nv pipeline and run on the GPU.
"""

import csv
import argparse
import concurrent.futures
import hashlib
import math
import mmap
import json
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
from gvn import bank_operations, build_plugins, repo, run, table_offset  # noqa: E402

from utils.path import log_dir  # noqa: E402


def check_captured_values(elf, expected, element):
    """Compare the values captured by BEMU with the host graph's boundary."""
    with elf.open("rb") as stream, mmap.mmap(
        stream.fileno(), 0, access=mmap.ACCESS_READ
    ) as image:
        header = struct.unpack_from("<16sHHIQQQIHHHHHH", image)
        sections = [
            struct.unpack_from("<IIQQQQIIQQ", image, header[6] + index * header[11])
            for index in range(header[12])
        ]
        strings = sections[header[13]]
        names = image[strings[4] : strings[4] + strings[5]]
        (check,) = [
            s
            for s in sections
            if names[s[0] :].split(b"\0", 1)[0] == b".bbsim.check_output"
        ]
        _, address, size = struct.unpack_from("<QQQ", image, check[4])
        (data,) = [
            s
            for s in sections
            if s[1] != 8 and s[3] <= address and address + size <= s[3] + s[5]
        ]
        offset = data[4] + address - data[3]
        actual = image[offset : offset + size]
    if len(expected) != size:
        raise ValueError(
            f"{elf.name}: captured {size} bytes, reference has {len(expected)}"
        )
    if element == "f32":
        largest = error = 0.0
        for (x,), (y,) in zip(
            struct.iter_unpack("<f", actual),
            struct.iter_unpack("<f", expected),
            strict=True,
        ):
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f"{elf.name}: nonfinite slice output")
            largest = max(largest, abs(y))
            error = max(error, abs(x - y))
        if error > 1e-5 * max(1.0, largest):
            raise ValueError(
                f"{elf.name}: FP32 slice error {error}, reference magnitude {largest}"
            )
    elif actual != expected:
        raise ValueError(f"{elf.name}: integer slice differs from reference")


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
    model = config / "models" / args.model
    harness = repo / "examples/models" / args.design / args.model / "gvn"
    out = config / "gvn" / args.model
    out.mkdir(parents=True, exist_ok=True)

    build_plugins()
    npu = out / "npu"

    # ===--- NPU side: checks and slices ---=== #
    # The recipe's tree (gvn/ of the recipe) checks the hart graph, then,
    # configured again, cuts one slice per check into batches of loop
    # iterations and lowers the slices and the whole graph.
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
    ids = re.findall(r'gvn\.check %\S+ id "([^"]+)"', (npu / "checks.mlir").read_text())
    print(f"{len(ids)} slices, checks at {', '.join(ids)}")

    # Every iteration of a slice is some stretch of the whole graph's
    # physical bank operations.
    whole = bank_operations((npu / "whole_physical.mlir").read_text())
    slices = []
    for index in range(len(ids)):
        directory = out / f"slice{index}"
        operations = bank_operations((directory / "slice_physical.mlir").read_text())
        if not any(
            whole[start : start + len(operations)] == operations
            for start in range(len(whole) - len(operations) + 1)
        ):
            sys.exit(
                f"slice {index}: its physical bank operations are no stretch of the whole graph's"
            )
        slices.append(json.loads((directory / "layout.json").read_text()))
    print("physical banks: every slice is a stretch of the whole graph")

    # ===--- GPU side: the checked values ---=== #
    # The checks, and the CPU computations the iterations read, computed by
    # stack/gvn/gpu with the entry and the model's host harness.
    ids = sorted(
        {*ids, *(entry["id"] for layout in slices for entry in layout["entries"])}
    )
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

    # ===--- One ELF per slice ---=== #
    # The values each slice reads, for the recipe's tree to embed: gvn_in
    # holds them, each 64-byte aligned; gvn_table their offsets (their shapes
    # are static) and the iteration; gvn_expect the slice's output.
    shapes = {value["id"]: value for value in exposed}
    for layout in slices:
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
        expected = values[layout["output"]["id"]]
        (directory / "gvn_expect.bin").write_bytes(expected)
        # A full slice must write every byte. Preserve untouched iterations
        # only when the harness deliberately runs part of a loop.
        initial = (
            bytes(byte ^ 0xFF for byte in expected)
            if layout["iteration_slot"] < 0
            else expected
        )
        (directory / "gvn_initial.bin").write_bytes(initial)
    run(
        [
            "cmake",
            "--build",
            recipe_build,
            "--target",
            f"{args.model}-gvn-elf",
            "--parallel",
            str(args.jobs),
        ]
    )

    # Debug runs seek coverage: one instance of each distinct slice (its
    # first iteration), slices compiled to the same object being one.
    # A slice runs once per iteration in the whole graph: the repeats of a
    # distinct slice are the iterations of every slice compiled to it.
    patterns, repeats = {}, {}
    for layout in slices:
        code = hashlib.file_digest(
            (out / f"slice{layout['index']}" / "slice.o").open("rb"), "sha256"
        ).hexdigest()
        patterns.setdefault(code, layout)
        repeats[patterns[code]["index"]] = (
            repeats.get(patterns[code]["index"], 0) + layout["iterations"]
        )
    instances = [(layout, 0) for layout in patterns.values()]
    layouts = {layout["index"]: layout for layout in slices}
    print(
        f"{len(instances)} instances: one of each of {len(patterns)} distinct slices of {len(slices)}"
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
    (logs / "slices.json").write_text(json.dumps(slices, indent=2) + "\n")
    bemu = config / "bin/bebop-bemu"

    def instance(job):
        layout, iteration = job
        index = layout["index"]
        name = f"{args.design}-{args.model}-gvn-slice{index}-iteration{iteration}"
        directory = out / "instances" / name
        directory.mkdir(parents=True, exist_ok=True)
        elf = directory / name
        shutil.copy2(out / f"slice{index}" / "slice.elf", elf)
        if layout["iteration_slot"] >= 0:
            with elf.open("r+b") as stream:
                stream.seek(table_offset(elf) + layout["iteration_slot"] * 8)
                stream.write(struct.pack("<q", iteration))
        log = logs / name
        log.mkdir(parents=True, exist_ok=True)
        begin = time.monotonic()
        result = subprocess.run(
            [bemu, "--elf", elf, "--log-dir", log, "--itrace", "--tool-profile"],
            cwd=log,
            capture_output=True,
            text=True,
        )
        seconds = time.monotonic() - begin
        (log / "run.log").write_text(result.stdout + result.stderr)
        if result.returncode == 0:
            check_captured_values(
                log / "reference.elf",
                (out / f"slice{index}" / "gvn_expect.bin").read_bytes(),
                layout["output"]["element"],
            )
            (log / "reference.elf").rename(log / name)
        else:
            shutil.copy2(elf, log / name)
        shutil.rmtree(directory)
        report = [
            line
            for line in result.stdout.splitlines()
            if line.startswith("Memory reference captured")
        ]
        if result.returncode == 0 and len(report) != 1:
            raise RuntimeError(
                f"{name}: expected one BEMU reference capture, got {report}"
            )
        if result.returncode == 0:
            with (log / "id-0/cpu-instructions.csv").open() as stream:
                cpu_cycles = sum(
                    int(row["modeled_active_cycles"]) for row in csv.DictReader(stream)
                )
            intervals = []
            with (log / "id-0/bdb.ndjson").open() as stream:
                for line in stream:
                    event = json.loads(line)
                    if event["type"] == "itrace" and "issue_cycle" in event:
                        start = event["issue_cycle"]
                        intervals.append((start, start + event["modeled_cycles"]))
            end, npu_cycles = 0, 0
            for start, stop in sorted(intervals):
                npu_cycles += max(0, stop - max(start, end))
                end = max(end, stop)
            (log / "performance.json").write_text(
                json.dumps(
                    {
                        "cpu_active_cycles": cpu_cycles,
                        "npu_active_union_cycles": npu_cycles,
                        "npu_finish_cycle": end,
                    },
                    indent=2,
                )
                + "\n"
            )
        timing = log / "trace/controller-0/thread-0/cycle/trace-9223372036854775807.txt"
        timing_text = timing.read_text()
        cycles = re.findall(r"^elapsed (\d+)$", timing_text, re.MULTILINE)
        elapsed_cycles = int(cycles[0]) if cycles else -1
        if result.returncode == 0:
            (start,) = re.findall(r"^start (\d+)$", timing_text, re.MULTILINE)
            # The guest trace samples its counter before its MMIO fence.
            # Include pending accelerator completion in the slice duration.
            elapsed_cycles = max(elapsed_cycles, end - int(start))
        return name, result.returncode, report, elapsed_cycles, seconds, index

    begin = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        runs = list(pool.map(instance, instances))
    elapsed = time.monotonic() - begin
    (out / "instances.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "instance": name,
                    "exit": code,
                    "cycles": cycles,
                    "seconds": round(seconds, 3),
                    "report": report,
                    # For estimates: the slice's ELF, the table slot of its
                    # iteration and the times it runs in the whole graph.
                    "elf": str(logs / name / name),
                    "reference_elf": str(logs / name / name),
                    "iteration_slot": layouts[index]["iteration_slot"],
                    "repeats": repeats[index],
                }
            )
            + "\n"
            for name, code, report, cycles, seconds, index in runs
        )
    )
    (logs / "instances.jsonl").write_bytes((out / "instances.jsonl").read_bytes())
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


if __name__ == "__main__":
    main()
