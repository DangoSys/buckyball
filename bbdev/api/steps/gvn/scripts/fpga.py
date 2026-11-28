import asyncio
import fcntl
import json
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

from steps.bebop.p2e.scripts.runtime_case import case_ready, trace_modes
from steps.workload.scripts.elf2hex import elf_to_hex
from utils.path import design_output_root, get_buckyball_path
from utils.run_step import run


def bitstream_of(target, request):
    """The bitstream an FPGA node runs on (the request's, or the device's
    latest completed build) and whether it carries DiffTest."""
    repo = Path(get_buckyball_path())
    output = Path(
        design_output_root(repo, target["design"], selection=target["target"])
    )
    if "bitstream" in request:
        bit = (repo / request["bitstream"]).resolve()
    else:
        cases = sorted(
            p
            for p in (output / "p2e").glob("*/fpgaCompDir/bitstream.bit")
            if case_ready(p)
        )
        if not cases:
            raise ValueError(f"No completed FPGA build under {output / 'p2e'}")
        bit = cases[-1].resolve()
    if not bit.is_relative_to(output.resolve() / "p2e"):
        raise ValueError(f"Bitstream must belong to {output}/p2e")
    if not case_ready(bit):
        raise ValueError(f"Incomplete FPGA case for {bit}")
    diff = bool(trace_modes(str(bit)) & {"btrace", "btrace_nb_v1"})
    if sum(part.is_dir() for part in bit.parent.glob("part_b*_f*")) > 1:
        raise ValueError("FPGA bench requires a single-FPGA case for the selected slot")
    return bit, diff


async def fpga_bench(ctx, target, node, request, selected, write, debug):
    """FPGA GVN (fpga-gvn: each distinct slice's reference ELF, its cycles
    times its repeats; with debug, each one's check against BEMU, pass or
    fail) or whole models (fpga-model)."""
    if not selected:
        return {"status": "skipped"}
    repo = Path(get_buckyball_path())
    output = Path(
        design_output_root(repo, target["design"], selection=target["target"])
    )
    bit, diff = bitstream_of(target, request)
    board = request.get("fpga-location", "0.A")
    hardware = {
        **target,
        "bitstream": str(bit),
        "fpga-location": board,
        "diff": diff,
    }
    node.update(bitstream=str(bit), fpga_location=board, diff=diff)
    gvn = node["id"] == "fpga-gvn"
    rows = []
    node["estimates" if gvn else "models"] = rows
    write()

    async def execute(product, elf=None):
        # VVAC creates this service directory file at cold load. The previous
        # host's P0 endpoint can outlive its process and strand the next CTB
        # initialization. Both the case and the board are locked here.
        services = bit.parent.parent / "vvacDir/runtimeDir/tbservice.toml"
        if services.is_file():
            with services.open("rb") as stream:
                registered = tomllib.load(stream)
            if registered.keys() - {"P0"}:
                raise RuntimeError(f"P2E services are still registered in {services}")
            services.unlink()
        if elf is None:
            args = {**hardware, **product}
            if diff:
                args["ddr-image"] = str(
                    output
                    / "workloads/image"
                    / f"{target['design']}-{product['model']}-linux-ddr.bin"
                )
            state, body = await run(ctx, "bebop.p2e.runworkload", args)
        else:
            with tempfile.TemporaryDirectory(dir=output / "workloads") as directory:
                name = f"{elf.name}-{Path(directory).name}"
                staged = Path(directory) / name
                staged.symlink_to(elf.resolve(strict=True))
                await asyncio.to_thread(
                    elf_to_hex,
                    staged,
                    Path(directory) / (name + ".hex"),
                    "riscv64-unknown-elf-objcopy",
                )
                args = {**hardware, "test": name}
                # A model whose parameters lie in its model region has its
                # DDR image beside its ELF, as BEMU loads it: the board takes
                # the image from where its data starts.
                ddr = elf.with_name(elf.name + "-ddr.bin")
                if ddr.is_file():
                    model = Path(directory) / "model.img"
                    with ddr.open("rb") as source, model.open("wb") as image:
                        start = os.lseek(source.fileno(), 0, os.SEEK_DATA)
                        source.seek(start)
                        shutil.copyfileobj(source, image, 1 << 24)
                    args["model-image"] = str(model)
                    args["model-offset"] = hex(start)
                state, body = await run(ctx, "bebop.p2e.runworkload", args)
        if state != "success":
            raise RuntimeError(f"P2E {state}: {body}")
        log = Path(body["log_dir"])
        timing = (log / "p2e_timing.log").read_text()
        (seconds,) = re.findall(r"Total P2E command: ([\d.]+) s", timing)
        return log, float(seconds)

    # The board is exclusive; the runtime also writes flags and Tcl into its case.
    with (repo / "log" / f"fpga-{board}.lock").open("a") as board_lock, (
        bit.parent.parent / "bench.lock"
    ).open("a") as case_lock:
        await asyncio.to_thread(fcntl.flock, board_lock, fcntl.LOCK_EX)
        await asyncio.to_thread(fcntl.flock, case_lock, fcntl.LOCK_EX)
        for model in selected:
            if gvn:
                row = {"model": model, "status": "running", "slices": []}
                rows.append(row)
                write()
                state, body = await run(
                    ctx, "gvn.bemu.debug", {**target, "model": model}
                )
                if state != "success":
                    raise RuntimeError(f"GVN preparation failed for {model}: {body}")
                manifest = Path(body["manifest"])
                row["reference_log"] = manifest.parent.name
                records = [
                    json.loads(line) for line in manifest.read_text().splitlines()
                ]
                if not records or any(
                    rec["exit"] or rec["repeats"] <= 0 or rec["cycles"] <= 0
                    for rec in records
                ):
                    raise ValueError(f"Incomplete GVN references for {model}")
                # debug: every distinct slice; perform: the heaviest ones
                # (BEMU cycles times repeats) until nine tenths of the
                # model's time, the rare rest at its BEMU cycles.
                rare = []
                if not debug:
                    weight = sum(rec["repeats"] * rec["cycles"] for rec in records)
                    heaviest, covered = [], 0
                    for rec in sorted(
                        records, key=lambda r: -r["repeats"] * r["cycles"]
                    ):
                        if covered >= 0.9 * weight:
                            break
                        heaviest.append(rec)
                        covered += rec["repeats"] * rec["cycles"]
                    rare = [rec for rec in records if rec not in heaviest]
                    records = heaviest
                for rec in records:
                    if debug:
                        # The reference exits nonzero when the FPGA's output
                        # differs from BEMU's.
                        try:
                            log, seconds = await execute({}, Path(rec["reference_elf"]))
                            row["slices"].append(
                                {
                                    "instance": rec["instance"],
                                    "exit": 0,
                                    "log": log.name,
                                }
                            )
                        except RuntimeError as error:
                            row["slices"].append(
                                {
                                    "instance": rec["instance"],
                                    "exit": 1,
                                    "error": str(error),
                                }
                            )
                        write()
                        continue
                    log, seconds = await execute({}, Path(rec["reference_elf"]))
                    (trace,) = log.glob(
                        "trace/controller-*/thread-*/cycle/trace-9223372036854775807.txt"
                    )
                    (span,) = re.findall(r"^elapsed (\d+)$", trace.read_text(), re.M)
                    cycles = int(span)
                    if cycles == 0:
                        raise ValueError(
                            f"Empty FPGA cycle span for {rec['instance']} in {log}"
                        )
                    row["slices"].append(
                        {
                            "instance": rec["instance"],
                            "elapsed_s": seconds,
                            "cycles": cycles,
                            "repeats": rec["repeats"],
                            "total": cycles * rec["repeats"],
                            "exit": 0,
                            "log": log.name,
                        }
                    )
                    write()
                if debug:
                    failed = sum(1 for item in row["slices"] if item["exit"])
                    row.update(
                        status="failure" if failed else "pass",
                        passed=len(row["slices"]) - failed,
                        failed=failed,
                    )
                    write()
                    continue
                rare_total = sum(rec["repeats"] * rec["cycles"] for rec in rare)
                row["rare"] = [
                    {
                        "instance": rec["instance"],
                        "repeats": rec["repeats"],
                        "bemu_cycles": rec["cycles"],
                    }
                    for rec in rare
                ]
                row["rare_bemu_cycles"] = rare_total
                total = sum(item["total"] for item in row["slices"]) + rare_total
                for item in row["slices"]:
                    item["share"] = item["total"] / total
                row.update(
                    status="pass",
                    cycles=total,
                    elapsed_s=round(
                        sum(item["elapsed_s"] for item in row["slices"]), 3
                    ),
                )
                write()
            else:
                for os_name in ("baremetal", "linux"):
                    row = {"model": model, "os": os_name, "status": "running"}
                    rows.append(row)
                    write()
                    try:
                        product = {"model": model, "os": os_name}
                        elf = None
                        if os_name == "linux":
                            state, body = await run(
                                ctx, "image.build", {**target, "model": model}
                            )
                            if state != "success":
                                raise RuntimeError(
                                    f"Linux image build failed for {model}: {body}"
                                )
                        else:
                            elf = (
                                output
                                / "workloads/models"
                                / f"{target['design']}-{model}-baremetal"
                            )
                        log, seconds = await execute(product, elf)
                        row["log"] = log.name
                        uart = "\n".join(
                            path.read_text()
                            for path in sorted(log.glob("uart_hart_*.log"))
                        )
                        spans = re.findall(r"Inference cycles: (\d+)", uart)
                        if len(spans) != 1 or int(spans[0]) <= 0:
                            raise ValueError(
                                f"Expected one positive inference cycle count in {log}"
                            )
                        cycles = int(spans[0])
                        check = re.search(
                            r"Output max diff: ([0-9.eE+-]+) \((\d+) elements, relative ([0-9.eE+-]+)\)",
                            uart,
                        )
                        if check:
                            row.update(
                                max_diff=float(check[1]),
                                elements=int(check[2]),
                                relative=float(check[3]),
                            )
                        row.update(status="pass", cycles=cycles, elapsed_s=seconds)
                    except (
                        ValueError,
                        RuntimeError,
                        OSError,
                        KeyError,
                        subprocess.SubprocessError,
                    ) as error:
                        row.update(status="failure", error=str(error))
                    write()
    return {
        "status": "pass" if all(row["status"] == "pass" for row in rows) else "failure"
    }
