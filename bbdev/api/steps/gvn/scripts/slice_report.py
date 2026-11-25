"""The report of one BEMU run of a GVN slice instance, in its own process so
instances report in parallel, as performance.json: the CPU cycles (the
per-PC profile) and the elapsed cycles of the slice (its trace span; the
readback is after it). The hart's clock is virtual: it does not count its
waiting, so the elapsed cycles beyond the CPU's are the NPU's alone.

  python3 slice_report.py <run directory> <verify bank> <instance name> \
      (reference | cycles)

reference (a run with instruction and memory traces): also freeze the
coherent readback as the BEMU reference (bank_output) and take the NPU active
cycles of the slice and of its readback from the instruction trace; the
traces are then removed, the reference ELF reproduces them.
"""

import csv
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bank_output  # noqa: E402

log, bank, name, mode = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
cpu = sum(
    int(row["modeled_active_cycles"])
    for path in log.glob("id-*/cpu-instructions.csv")
    for row in csv.DictReader(path.open())
)
(trace,) = log.glob("trace/**/trace-9223372036854775807.txt")
elapsed = int(re.search(r"^elapsed (\d+)$", trace.read_text(), re.M)[1])
report = {
    "cpu_active_cycles_including_setup_and_readback": cpu,
    "inference_elapsed_cycles": elapsed,
}
if mode == "reference":
    bank_output.capture(log)
    checked = bank_output.check(log, log)
    shutil.copy2(log / "reference.elf", log / name)
    spans = [[], []]
    # CPU cycle markers and Ant issue cycles are different modeled clocks.
    # The readback starts with its bank allocation, immediately before its
    # first MVIN. Partition instructions at that boundary, not CPU time.
    # Instruction indices are per thread: the boundary is the readback
    # thread's; every instruction of the other threads is the slice's.
    first_readback = min(command["inst"] for command in checked["commands"]) - 1
    (reader,) = {command["hart"] for command in checked["commands"]}
    for path in log.glob("id-*/ant-*/bdb.ndjson"):
        events = [json.loads(line) for line in path.open()]
        # The thread's hart, which its memory events name.
        harts = {event["hart_id"] for event in events if "hart_id" in event}
        for event in events:
            if event["type"] != "itrace" or "issue_cycle" not in event:
                continue
            readback = reader in harts and event["event_index"] >= first_readback
            if reader in harts and event["event_index"] == first_readback:
                assert event["funct"] == "0x20" and int(event["rs1"], 16) == bank
            spans[readback].append(
                (event["issue_cycle"], event["issue_cycle"] + event["modeled_cycles"])
            )
    active = []
    for phase in spans:
        end = cycles = 0
        for issue, finish in sorted(phase):
            cycles += max(0, finish - max(issue, end))
            end = max(end, finish)
        active.append(cycles)
    report["npu_active_union_cycles"] = active[0]
    report["verification_npu_active_cycles"] = active[1]
    report["npu_partition"] = "instruction boundary before coherent readback"
    for path in log.glob("id-*/ant-*/bdb.ndjson"):
        path.unlink()
else:
    shutil.copy2(log / "slice.elf", log / name)
(log / "performance.json").write_text(json.dumps(report, indent=2) + "\n")
# The per-PC profiles are summed; a rerun of the ELF gives them again.
for path in log.glob("id-*/*.csv"):
    path.unlink()
for path in log.glob("id-*/ant-*/*.csv"):
    path.unlink()
print(elapsed)
