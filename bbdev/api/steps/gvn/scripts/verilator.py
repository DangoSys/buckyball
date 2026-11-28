"""A rough end-to-end NPU cycle count of a model from its GVN slices: every
distinct slice of the debug run (one instance each, instances.jsonl of the
design's driver) runs once, cold, on Verilator; the estimate is the sum of
each slice's Verilator cycles times the times it runs in the whole model.
Host CPU work between launches is not counted, nor slices that fail or run
past thirty minutes on Verilator (listed with their exit, -1 for a timeout).

  python3 verilator.py <gvn output> <Verilator simulator> <log dir> <jobs> (turn|beside) (perform|debug)

turn: in the host's turn (utils/verilator_turn.py); beside: beside a turn
that leaves the cores (a correctness regression of one model at a time),
at the width given.

perform writes verilator.json (the cycles, the simulation time, and per slice
its repeats, its cycles and its share) to the log dir and the GVN output;
debug runs each distinct slice's reference ELF (the driver's reference_elf,
BEMU's output captured in it) and writes verilator-debug.json, each slice's
exit (0 when its RTL output equals BEMU's).
"""

import concurrent.futures
import contextlib
import hashlib
import json
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from gvn import table_offset  # noqa: E402
import bank_output  # noqa: E402
from utils.verilator_turn import verilator_turn  # noqa: E402

out, simulator, logs, jobs, turn, mode = (
    Path(sys.argv[1]),
    Path(sys.argv[2]),
    Path(sys.argv[3]),
    int(sys.argv[4]),
    sys.argv[5],
    sys.argv[6],
)
if turn not in ("turn", "beside"):
    sys.exit(f"the fifth argument is turn or beside, not {turn}")
# perform: every distinct slice's cycles, the estimate; debug: every distinct
# slice's reference ELF (its output checked against the BEMU reference on
# exit), pass or fail.
if mode not in ("perform", "debug"):
    sys.exit(f"the sixth argument is perform or debug, not {mode}")
# Run the shortest BEMU slices first. Every selected slice gets an actual RTL measurement or exit.
distinct = [
    json.loads(line) for line in (out / "instances.jsonl").read_text().splitlines()
]
# Which slices run. debug seeks coverage: every distinct slice once, however
# rarely it runs. perform weighs by occurrence: the slices that hold the
# model's time (on BEMU: cycles times repeats), the heaviest first, until they
# cover nine tenths of it; the rare rest is counted at its BEMU cycles.
if mode == "debug":
    selected, rare = distinct, []
else:
    weight = sum(record["repeats"] * record["cycles"] for record in distinct)
    selected, covered = [], 0
    for record in sorted(distinct, key=lambda r: -r["repeats"] * r["cycles"]):
        if covered >= 0.9 * weight:
            break
        selected.append(record)
        covered += record["repeats"] * record["cycles"]
    rare = [record for record in distinct if record not in selected]
records = sorted(selected, key=lambda record: record["cycles"])
# The earlier estimates of this model (the same name after the time stamp),
# newest first: a slice one of them measured on the same ELF (elf.sha256
# beside its trace) is not simulated again.
earlier = sorted(
    (
        directory
        for directory in logs.parent.glob(f"*-{logs.name[20:]}")
        if directory != logs and directory.name[:19] < logs.name[:19]
    ),
    reverse=True,
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


simulator_digest = digest(simulator)
(logs / "simulator.sha256").write_text(simulator_digest + "\n")
earlier = [
    path
    for path in earlier
    if (path / "simulator.sha256").is_file()
    and (path / "simulator.sha256").read_text().strip() == simulator_digest
]
debug_times = {}
captures = []
if mode == "debug":
    for directory in earlier:
        report = directory / "verilator-debug.json"
        if report.is_file():
            debug_times[directory] = {
                r["instance"]: r["elapsed_s"]
                for r in json.loads(report.read_text())["slices"]
                if r["exit"] == 0 and "elapsed_s" in r
            }
    # A perf run of a capture ELF writes the entire RTL output into its
    # reference ELF. An exact match with the checked BEMU ELF proves both
    # program identity and output equality.
    suffix = logs.name[20:]
    if "-gvn-verilator-debug-" in suffix:
        perf_suffix = suffix.replace("-gvn-verilator-debug-", "-gvn-verilator-", 1)
        for directory in sorted(logs.parent.glob(f"*-{perf_suffix}"), reverse=True):
            identity = directory / "simulator.sha256"
            if identity.is_file() and identity.read_text().strip() == simulator_digest:
                captures.append(directory)


def terminal_cycles(log):
    traces = list(
        log.glob("trace/controller-*/thread-*/cycle/trace-9223372036854775807.txt")
    )
    if len(traces) != 1:
        return None
    spans = re.findall(r"^elapsed (\d+)$", traces[0].read_text(), re.M)
    return int(spans[0]) if len(spans) == 1 and int(spans[0]) > 0 else None


def measured_before(record, elf_digest):
    """The cycles an earlier estimate traced for this slice, and where."""
    for directory in earlier:
        instance = directory / record["instance"]
        if (
            not (instance / "elf.sha256").exists()
            or (instance / "elf.sha256").read_text().strip() != elf_digest
        ):
            continue
        # The slice ends its cycle trace before readback and simulator shutdown.
        # A timeout during either can leave a positive span in an unfinished run.
        log = instance / "run.log"
        if not log.is_file() or not (instance / "rtl-trace-summary.json").is_file():
            continue
        output = log.read_text(errors="replace")
        if (
            not re.search(
                r"^\[SCU\] sim_exit called: hart_id=\d+, exit_code=0$", output, re.M
            )
            or "Verilator timed out after" in output
            or re.search(r"^Error:", output, re.M)
        ):
            continue
        elapsed = None
        if mode == "debug":
            receipt = instance / "debug-receipt.json"
            elapsed = (
                json.loads(receipt.read_text())["elapsed_s"]
                if receipt.is_file()
                else debug_times.get(directory, {}).get(record["instance"])
            )
            if elapsed is None:
                continue
            proof = instance / "bank-output-check.json"
            if proof.is_file():
                check = json.loads(proof.read_text())
                if not check["passed"] or check["bytes"] <= 0:
                    continue
            elif not re.search(
                r"^Memory check passed: [1-9]\d* bytes at 0x[0-9a-fA-F]+$", output, re.M
            ):
                continue
        cycles = terminal_cycles(instance)
        if cycles is not None:
            return cycles, directory.name, elapsed
    for directory in captures:
        instance = directory / record["instance"]
        captured = instance / "reference.elf"
        log = instance / "run.log"
        if (
            not captured.is_file()
            or not log.is_file()
            or not (instance / "rtl-trace-summary.json").is_file()
        ):
            continue
        output = log.read_text(errors="replace")
        if (
            "Verilator timed out after" in output
            or re.search(r"^Error:", output, re.M)
            or not re.search(
                r"^\[SCU\] sim_exit called: hart_id=\d+, exit_code=0$", output, re.M
            )
            or not re.search(
                r"^Memory reference captured: [1-9]\d* bytes at 0x[0-9a-fA-F]+$",
                output,
                re.M,
            )
            or digest(captured) != elf_digest
        ):
            continue
        cycles = terminal_cycles(instance)
        if cycles is not None:
            record["verified_capture"] = str(captured)
            receipt = instance / "timing.json"
            elapsed = (
                json.loads(receipt.read_text())["elapsed_s"]
                if receipt.is_file()
                else None
            )
            return cycles, directory.name, elapsed
    return None


def prepare(record):
    """The slice's ELF in its log directory (its first iteration, as the
    debug run's instance) and its digest; the cycles an earlier estimate
    measured on that same ELF, if one did."""
    log = logs / record["instance"]
    log.mkdir(parents=True)
    elf = log / record["instance"]
    source = record.get(
        "verilator_reference_elf",
        record["elf" if mode == "perform" else "reference_elf"],
    )
    shutil.copy2(source, elf)
    if record["iteration_slot"] >= 0:
        with elf.open("r+b") as stream:
            stream.seek(table_offset(elf) + record["iteration_slot"] * 8)
            stream.write(struct.pack("<q", 0))
    elf_digest = digest(elf)
    (log / "elf.sha256").write_text(elf_digest + "\n")
    before = measured_before(record, elf_digest)
    if before:
        elf.unlink()
        record["reused"], record["reused_cycles"] = before[1], before[0]
        record["elapsed_s"] = before[2]


def run(record):
    """The slice's cycles on Verilator: the span its own cycle trace takes."""
    log = logs / record["instance"]
    elf = log / record["instance"]
    if "reused" in record:
        return record, 0, record["reused_cycles"]
    begin = time.monotonic()
    command = [simulator, f"--elf={elf}", f"--log-dir={log}", "--no-wave"]
    reference = Path(record["reference_elf"]).parent if mode == "debug" else None
    bank_check = (
        reference is not None and (reference / "bank-output-capture.json").is_file()
    )
    if bank_check:
        command.extend(["--diff", "--itrace", "--mtrace"])
    # A first estimate need not be exact: a slice past thirty minutes on
    # Verilator (the limit of one Verilator run) is left out (exit -1), not
    # waited for.
    try:
        result = subprocess.run(
            command,
            cwd=log,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except subprocess.TimeoutExpired as error:
        with (log / "run.log").open("wb") as stream:
            stream.write(error.stdout or b"")
            stream.write(error.stderr or b"")
            stream.write(f"\nVerilator timed out after {error.timeout}s\n".encode())
        elf.unlink()
        return record, -1, None
    finally:
        record["elapsed_s"] = round(time.monotonic() - begin, 3)
    error = (
        f"Error: Verilator exited with status {result.returncode}\n"
        if result.returncode
        else ""
    )
    (log / "run.log").write_text(error + result.stdout + result.stderr)
    if bank_check and not result.returncode:
        try:
            bank_output.check(reference, log, frozen_elf=elf)
        except (ValueError, KeyError, AssertionError) as error:
            with (log / "run.log").open("a") as stream:
                stream.write(f"\nError: {error}\n")
            elf.unlink()
            return record, 1, None
    elf.unlink()
    if result.returncode:
        return record, result.returncode, None
    if (
        mode == "debug"
        and not bank_check
        and not re.search(
            r"^Memory check passed: [1-9]\d* bytes at 0x[0-9a-fA-F]+$",
            result.stdout + result.stderr,
            re.MULTILINE,
        )
    ):
        with (log / "run.log").open("a") as stream:
            stream.write("\nError: No BEMU reference comparison completed on RTL\n")
        return record, 1, None
    # The slice's own span, trace id INT64_MAX (the readback ends it); the
    # graphs inside the slice trace under ids of their own. It is traced on
    # the controller and thread the slice runs on, which differ by design.
    cycles = terminal_cycles(log)
    if cycles is None:
        with (log / "run.log").open("a") as stream:
            stream.write("\nError: Expected one positive terminal RTL cycle span\n")
        return record, 1, None
    if mode == "debug":
        # A long model can be interrupted after this slice passed.
        receipt = log / "debug-receipt.tmp"
        receipt.write_text(
            json.dumps({"cycles": cycles, "elapsed_s": record["elapsed_s"]}) + "\n"
        )
        receipt.replace(log / "debug-receipt.json")
    else:
        (log / "timing.json").write_text(
            json.dumps({"elapsed_s": record["elapsed_s"]}) + "\n"
        )
    return record, 0, cycles


# The simulation time: the slices on Verilator, side by side, in the host's
# turn (utils/verilator_turn.py): the estimates of every design, the tool
# bench and the timeline's Verilator regressions take turns, first come first
# served, so a turn has the host to itself.
for record in records:
    prepare(record)
# The host's turn only when something is left to simulate: a rerun whose
# slices were all measured before takes none.
with (
    verilator_turn(Path(__file__).resolve().parents[5])
    if turn == "turn" and any("reused" not in record for record in records)
    else contextlib.nullcontext()
):
    begin = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        runs = list(pool.map(run, records))
    seconds = round(time.monotonic() - begin, 1)

if mode == "debug":
    # Each distinct slice's reference on RTL: its exit says whether its
    # output equals BEMU's (0), differs or hangs (-1, the time out).
    checks = {
        "seconds": seconds,
        "slices": [
            {
                "instance": record["instance"],
                "exit": code,
                "elapsed_s": record["elapsed_s"],
                **({"reused": record["reused"]} if "reused" in record else {}),
                **(
                    {"verified_capture": record["verified_capture"]}
                    if "verified_capture" in record
                    else {}
                ),
            }
            for record, code, _ in runs
        ],
        "passed": sum(1 for _, code, _ in runs if code == 0),
        "failed": sum(1 for _, code, _ in runs if code != 0),
    }
    text = json.dumps(checks, indent=2) + "\n"
    (logs / "verilator-debug.json").write_text(text)
    (out / "verilator-debug.json").write_text(text)
    print(
        f"debug: {checks['passed']} of {len(runs)} distinct slices match BEMU on"
        f" Verilator; {logs / 'verilator-debug.json'}"
    )
    for record, code, _ in runs:
        if code:
            print(f"  {record['instance']}: exit {code}")
    sys.exit(0 if not checks["failed"] else 1)

measured_total = sum(
    record["repeats"] * cycles for record, code, cycles in runs if not code
)
# The rare slices, at their cycles on BEMU.
rare_total = sum(record["repeats"] * record["cycles"] for record in rare)
total = measured_total + rare_total
slices = sorted(
    (
        {
            "instance": record["instance"],
            "repeats": record["repeats"],
            "exit": code,
            "cycles": cycles,
            # The earlier estimate whose measurement this is, if one.
            **({"reused": record["reused"]} if "reused" in record else {}),
            "total": None if code else record["repeats"] * cycles,
            "share": None if code else record["repeats"] * cycles / total,
        }
        for record, code, cycles in runs
    ),
    key=lambda s: -(s["total"] or 0),
)
# What is not measured, apart from what is: the slices that timed out, that
# were not run, that failed, and their cycles on BEMU times their repeats.
unmeasured = {
    kind: [record for record, code, _ in runs if test(code)]
    for kind, test in (
        ("timed_out", lambda code: code == -1),
        ("not_run", lambda code: code == -2),
        ("failed", lambda code: code not in (0, -1, -2)),
    )
}
estimate = {
    "cycles": total,
    "measured_cycles": measured_total,
    # The slices not run for their rarity, and their BEMU cycles in the total.
    "rare": len(rare),
    "rare_bemu_cycles": rare_total,
    "rare_slices": [
        {
            "instance": record["instance"],
            "repeats": record["repeats"],
            "bemu_cycles": record["cycles"],
        }
        for record in rare
    ],
    "seconds": seconds,
    "slices": slices,
    "failed": sum(1 for s in slices if s["exit"]),
    "measured": sum(1 for s in slices if not s["exit"]),
    **{kind: len(group) for kind, group in unmeasured.items()},
    "unmeasured_bemu_cycles": {
        kind: sum(record["repeats"] * record["cycles"] for record in group)
        for kind, group in unmeasured.items()
    },
}
text = json.dumps(estimate, indent=2) + "\n"
(logs / "verilator.json").write_text(text)
(out / "verilator.json").write_text(text)
print(
    f"estimate: {total} NPU cycles over {len(slices)} distinct slices"
    f" ({estimate['measured']} measured, {estimate['rare']} rare at BEMU cycles"
    f" ({rare_total}), {estimate['timed_out']} timed out,"
    f" {estimate['not_run']} not run, {estimate['failed']} failed otherwise;"
    f" unmeasured BEMU cycles {estimate['unmeasured_bemu_cycles']});"
    f" {logs / 'verilator.json'}"
)
for s in slices[:10]:
    print(
        f"  {s['instance']}: {s['repeats']} x {s['cycles']} = {s['total']}"
        f" ({100 * (s['share'] or 0):.1f}%)"
    )
sys.exit(0 if slices and all(s["exit"] == 0 and s["cycles"] > 0 for s in slices) else 1)
