import argparse
from array import array
from collections import deque
import csv
import heapq
import json
import math
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

repo = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(repo))
from stack.scripts.design import load

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from utils.path import get_buckyball_path, run_log_dir


class Replay:
    def __init__(self, thread, virtual_banks, calibration):
        self.capacity = thread.mem.bank.num
        self.depth = thread.frontend.rob_entries
        self.alias_base = virtual_banks + (2 if thread.gpdomain.enable else 0)
        self.bids = {
            i.funct7: ball.id
            for ball in thread.balldomain.balls
            for i in ball.instructions
        }
        self.calibration = calibration
        self.now = 0
        self.frontier = deque()
        self.pending = []
        self.tail = 0
        self.next_alloc = 0
        self.next_issue = 0
        self.mapping = {}
        self.current_alias = list(range(self.alias_base))
        self.alias_busy = set()
        self.alias_live = set()
        self.alias_readers = [0] * (self.alias_base + self.depth)
        self.busy_mask = 0
        self.busy_count = 0
        self.count = 0
        self.events = [
            array("Q", [0]),
            array("B", [0]),
            array("B", [0]),
            array("B", [0]),
        ]
        self.commands = 0
        self.frontend_stalls = 0

    def record(self):
        values = (self.count, len(self.alias_live), self.busy_count)
        assert (
            values[0] <= self.capacity
            and values[1] <= self.depth
            and values[2] <= values[0]
        )
        if values != tuple(a[-1] for a in self.events[1:]):
            self.events[0].append(self.now)
            for column, value in zip(self.events[1:], values):
                column.append(value)

    def enqueue(self, event):
        f, a, b = event["f"], event["a"], event["b"]
        mode = f >> 4
        reads = [a & 1023] if mode in (1, 3, 4) or (f == 32 and b & 4096) else []
        if mode == 4:
            reads.append((a >> 10) & 1023)
        write = (
            ((a & 1023) if f == 32 and not b & 4096 else (a >> 20) & 1023)
            if mode in (2, 3, 4) and f != 35
            else None
        )
        assert all(v < self.alias_base for v in reads) and (
            write is None or write < self.alias_base
        )
        config = f == 32
        bid = self.bids[f] if f in self.bids else None
        uses = set(reads)
        if write is not None:
            uses.add(write)
        read_mask = sum(1 << v for v in set(reads))
        write_mask = 0 if write is None else 1 << write
        mask = read_mask | write_mask
        deps = []
        for older in self.frontier:
            conflict = bool(
                read_mask & older["write_mask"] or write_mask & older["mask"]
            )
            config_conflict = (
                config
                and older["config"]
                or (config or older["config"])
                and bool(mask & older["mask"])
            )
            if config_conflict:
                deps.append((older, "committed"))
            elif conflict or bid is not None and bid == older["bid"]:
                deps.append((older, "done"))
        alias = self.alias_base + self.tail if write is not None else None
        renamed_reads = [self.current_alias[v] for v in reads]
        entry = {
            "f": f,
            "a": a,
            "b": b,
            "base": event["base"],
            "reads": reads,
            "write": write,
            "writes": set() if write is None else {write},
            "uses": uses,
            "config": config,
            "bid": bid,
            "alias": alias,
            "renamed_reads": renamed_reads,
            "deps": deps,
            "issued": False,
            "done": False,
            "committed": False,
            "allocated": self.now,
            "physical": [],
            "number": self.commands,
            "mask": mask,
            "write_mask": write_mask,
        }
        if alias is not None:
            assert alias not in self.alias_live
            self.current_alias[write] = alias
            self.alias_live.add(alias)
            self.alias_busy.add(alias)
        for alias_read in renamed_reads:
            self.alias_readers[alias_read] += 1
        self.frontier.append(entry)
        self.tail = (self.tail + 1) % self.depth
        self.next_alloc = self.now + 1
        self.commands += 1

    def advance(self, incoming, arrival):
        while True:
            while self.pending and self.pending[0][0] <= self.now:
                _, _, entry = heapq.heappop(self.pending)
                entry["done"] = True
                entry["completed"] = self.now
                self.busy_mask &= ~entry["mask"]
                self.busy_count -= len(entry["physical"])
                for alias_read in entry["renamed_reads"]:
                    self.alias_readers[alias_read] -= 1
                if entry["alias"] is not None:
                    self.alias_busy.remove(entry["alias"])
            committed = False
            while (
                self.frontier
                and self.frontier[0]["done"]
                and self.frontier[0]["completed"] < self.now
            ):
                entry = self.frontier.popleft()
                entry["committed"] = True
                committed = True
                if entry["alias"] is not None:
                    self.alias_live.remove(entry["alias"])
                    if self.current_alias[entry["write"]] == entry["alias"]:
                        self.current_alias[entry["write"]] = entry["write"]
                if entry["config"]:
                    v, b = entry["a"] & 1023, entry["b"]
                    if b & 4096:
                        target = (entry["a"] >> 20) & 1023
                        self.mapping[target] = self.mapping.get(
                            target, []
                        ) + self.mapping.pop(v)
                    else:
                        self.mapping.pop(v, None)
                        if b & 1024:
                            groups = (b >> 5) & 31
                            groups = groups if groups else self.capacity
                            occupied = {p for ps in self.mapping.values() for p in ps}
                            available = [
                                p for p in range(self.capacity) if p not in occupied
                            ]
                            assert len(available) >= groups
                            self.mapping[v] = available[:groups]
                    self.count = sum(map(len, self.mapping.values()))
            if self.now >= self.next_issue:
                for entry in self.frontier:
                    if (
                        entry["issued"]
                        or entry["allocated"] >= self.now
                        or entry["mask"] & self.busy_mask
                    ):
                        continue
                    if any(not dep[field] for dep, field in entry["deps"]) or any(
                        a in self.alias_busy for a in entry["renamed_reads"]
                    ):
                        continue
                    entry["issued"] = True
                    entry["deps"] = []
                    entry["physical"] = [
                        p
                        for v in entry["uses"]
                        if not entry["config"]
                        for p in self.mapping[v]
                    ]
                    self.busy_mask |= entry["mask"]
                    self.busy_count += len(entry["physical"])
                    f, a, b = entry["f"], entry["a"], entry["b"]
                    factor = self.calibration[str(f)]
                    base = entry["base"]
                    if f in (16, 33, 49, 69, 70, 72) and not (
                        f in (16, 33) and b >> 63
                    ):
                        v = (
                            (a & 1023)
                            if f in (16, 49, 69, 70, 72)
                            else (a >> 20) & 1023
                        )
                        base *= len(self.mapping[v])
                    if (
                        factor.get("model")
                        == "effective_rows*K + bias_rows + writeback_rows + startup"
                    ):
                        rows, k = b & 4095, a >> 30
                        features = (
                            (7.125 if rows == 1 else rows) * k,
                            ((b >> 24) & 1) * rows,
                            ((b >> 25) & 1) * rows,
                            1,
                        )
                        duration = sum(
                            x * y for x, y in zip(features, factor["coefficients"])
                        )
                    else:
                        duration = base * factor["scale"]
                    assert duration > 0
                    entry["issue_cycle"] = self.now
                    completion = self.now + math.ceil(duration)
                    heapq.heappush(self.pending, (completion, entry["number"], entry))
                    self.next_issue = self.now + 1
                    break
            # The RTL forbids allocating on a commit cycle and reusing an alias
            # still referenced by any incomplete entry.
            alias = self.alias_base + self.tail
            alias_referenced = (
                alias in self.alias_busy or self.alias_readers[alias] != 0
            )
            if (
                incoming is not None
                and arrival <= self.now
                and not committed
                and len(self.frontier) < self.depth
                and self.now >= self.next_alloc
                and not alias_referenced
            ):
                self.enqueue(incoming)
                self.record()
                return
            self.record()
            if incoming is None and not self.frontier:
                return
            candidates = []
            if incoming is not None and arrival > self.now:
                candidates.append(arrival)
            if self.pending:
                candidates.append(self.pending[0][0])
            if self.frontier and self.frontier[0]["done"]:
                candidates.append(max(self.now + 1, self.frontier[0]["completed"] + 1))
            if any(
                not e["issued"]
                and (not any(not dep[field] for dep, field in e["deps"]))
                and not any(a in self.alias_busy for a in e["renamed_reads"])
                and not e["mask"] & self.busy_mask
                for e in self.frontier
            ):
                candidates.append(max(self.now + 1, self.next_issue))
            if committed:
                candidates.append(self.now + 1)
            if (
                incoming is not None
                and len(self.frontier) < self.depth
                and not alias_referenced
                and self.next_alloc > self.now
            ):
                candidates.append(self.next_alloc)
            assert candidates, "Replay deadlock"
            self.now = min(t for t in candidates if t > self.now)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-pb", type=Path, required=True)
    parser.add_argument("--thread-index", type=int, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_log = run_log_dir(get_buckyball_path(), args.run)
    assert all(
        p.is_absolute() for p in [args.design_pb, args.calibration, args.output_dir]
    )
    design = load(args.design_pb)
    thread = design.threads[args.thread_index]
    core = next(t for t in design.device.cores if args.thread_index in t.thread_indices)
    model = Replay(
        thread, core.virtual_bank_count, json.loads(args.calibration.read_text())
    )
    started = time.monotonic()
    first_cycle = None
    with (run_log / "bdb.ndjson").open() as f:
        for line in f:
            e = json.loads(line)
            if e["type"] != "itrace":
                continue
            if first_cycle is None:
                first_cycle = e["issue_cycle"]
            funct = int(e["funct"], 16)
            if funct == 1:
                model.advance(None, 0)
                continue
            base = e["modeled_cycles"]
            if base is None:
                raise ValueError("This kernel has no validated timing model")
            model.advance(
                {
                    "f": funct,
                    "a": int(e["rs1"], 16),
                    "b": int(e["rs2"], 16),
                    "base": base,
                },
                e["issue_cycle"] - first_cycle,
            )
    model.advance(None, 0)
    model.events[0].append(model.now)
    for a in model.events[1:]:
        a.append(a[-1])
    args.output_dir.mkdir(exist_ok=True, parents=True)
    values = np.array(model.events, dtype=np.uint64)
    np.savez_compressed(
        args.output_dir / "bank-replay-events.npz",
        cycle=values[0],
        allocated_physical_banks=values[1],
        live_bat_aliases=values[2],
        busy_physical_banks=values[3],
    )
    result = {
        "method": (
            "Event-driven replay of architectural-bank dependencies before the DMA issue-order "
            "fix, BAT alias lifetime and MSET physical mappings. Completion latencies are RTL- "
            "calibrated estimates; BEMU CPU issue times are arrival constraints. Not cycle- "
            "accurate RTL or FPGA measurement."
        ),
        "limitations": [
            (
                "The four-slot thread admission interface is not modeled; ROB occupancy may exceed "
                "what the current issue path can supply."
            ),
            "The per-producer DMA issue-order dependency added after the MVOUT deadlock diagnosis is not modeled.",
            "Issue transport/endpoint backpressure is not modeled cycle by cycle.",
            "Latency calibration extrapolates outside measured shapes.",
            "Alias count is not a physical-bank count.",
        ],
        "log_dir": str(run_log),
        "design_pb": str(args.design_pb),
        "thread_index": args.thread_index,
        "rob_entries": model.depth,
        "physical_banks": model.capacity,
        "bank_bytes": thread.mem.bank.width // 8 * thread.mem.bank.entries,
        "instructions": model.commands,
        "estimated_cycles": model.now,
        "offline_seconds": time.monotonic() - started,
        "physical_allocation_peak": int(values[1].max()),
        "bat_alias_peak": int(values[2].max()),
        "busy_physical_peak": int(values[3].max()),
        "final_physical_banks": model.count,
        "final_bat_aliases": len(model.alias_live),
    }
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), constrained_layout=True)
    for ax, y, label, capacity, color in zip(
        axes,
        values[1:],
        [
            "Mapped physical data banks",
            "Live BAT aliases",
            "In-flight reserved physical banks",
        ],
        [model.capacity, model.depth, model.capacity],
        ["#2563eb", "#d97706", "#0d9488"],
    ):
        ax.step(values[0] / 1e6, y, where="post", linewidth=0.65, color=color)
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(0, capacity + 1)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Estimated replay cycles (million; not FPGA time)")
    fig.savefig(args.output_dir / "bank-replay.png", dpi=180)
    fig.savefig(args.output_dir / "bank-replay.pdf")
    print(json.dumps(result, indent=2), flush=True)
