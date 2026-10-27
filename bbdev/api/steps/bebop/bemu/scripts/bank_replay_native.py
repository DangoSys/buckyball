import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from stack.scripts.design import load

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from utils.path import get_buckyball_path, run_log_dir


@njit(cache=True)
def simulate(
    commands, capacity, depth, alias_base, bids, scales, shape_model, coefficients
):
    valid = np.zeros(depth, np.bool_)
    issued = np.zeros(depth, np.bool_)
    done = np.zeros(depth, np.bool_)
    config = np.zeros(depth, np.bool_)
    function = np.zeros(depth, np.int64)
    operand1 = np.zeros(depth, np.uint64)
    operand2 = np.zeros(depth, np.uint64)
    base_cycles = np.zeros(depth, np.float64)
    write_bank = np.full(depth, -1, np.int64)
    read0 = np.full(depth, -1, np.int64)
    read1 = np.full(depth, -1, np.int64)
    uses = np.zeros(depth, np.uint64)
    writes = np.zeros(depth, np.uint64)
    dependencies = np.zeros(depth, np.uint64)
    config_dependencies = np.zeros(depth, np.uint64)
    allocated = np.zeros(depth, np.int64)
    completion = np.zeros(depth, np.int64)
    physical_count = np.zeros(depth, np.int64)
    current_alias = np.arange(alias_base, dtype=np.int64)
    alias_busy = np.zeros(depth, np.bool_)
    alias_live = np.zeros(depth, np.bool_)
    alias_readers = np.zeros(alias_base + depth, np.int64)
    owners = np.full(capacity, -1, np.int64)
    groups = np.zeros(alias_base, np.int64)
    n = len(commands)
    times = np.zeros(n * 4 + 2, np.uint64)
    mapped = np.zeros(n * 4 + 2, np.uint8)
    aliases = np.zeros(n * 4 + 2, np.uint8)
    reserved = np.zeros(n * 4 + 2, np.uint8)
    points = 1
    now = 0
    head = 0
    tail = 0
    live_entries = 0
    mapped_count = 0
    alias_count = 0
    busy_count = 0
    busy_mask = np.uint64(0)
    next_alloc = 0
    next_issue = 0
    index = 0
    while True:
        completed_mask = np.uint64(0)
        for slot in range(depth):
            if (
                valid[slot]
                and issued[slot]
                and not done[slot]
                and completion[slot] <= now
            ):
                done[slot] = True
                completed_mask |= np.uint64(1) << np.uint64(slot)
                busy_mask &= ~uses[slot]
                busy_count -= physical_count[slot]
                if write_bank[slot] >= 0:
                    alias_busy[slot] = False
                if read0[slot] >= 0:
                    alias_readers[read0[slot]] -= 1
                if read1[slot] >= 0:
                    alias_readers[read1[slot]] -= 1
        committed = False
        committed_mask = np.uint64(0)
        while live_entries and done[head] and completion[head] < now:
            slot = head
            valid[slot] = False
            live_entries -= 1
            committed = True
            committed_mask |= np.uint64(1) << np.uint64(slot)
            if write_bank[slot] >= 0:
                alias_live[slot] = False
                alias_count -= 1
                if current_alias[write_bank[slot]] == alias_base + slot:
                    current_alias[write_bank[slot]] = write_bank[slot]
            if config[slot]:
                v = np.int64(operand1[slot] & np.uint64(1023))
                b = operand2[slot]
                if b & np.uint64(4096):
                    target = np.int64(
                        (operand1[slot] >> np.uint64(20)) & np.uint64(1023)
                    )
                    for p in range(capacity):
                        if owners[p] == v:
                            owners[p] = target
                    groups[target] += groups[v]
                    groups[v] = 0
                else:
                    mapped_count -= groups[v]
                    groups[v] = 0
                    for p in range(capacity):
                        if owners[p] == v:
                            owners[p] = -1
                    if b & np.uint64(1024):
                        count = np.int64((b >> np.uint64(5)) & np.uint64(31))
                        if count == 0:
                            count = capacity
                        for p in range(capacity):
                            if owners[p] < 0 and count:
                                owners[p] = v
                                count -= 1
                                groups[v] += 1
                                mapped_count += 1
                        assert count == 0
            head = (head + 1) % depth
        for slot in range(depth):
            dependencies[slot] &= ~(completed_mask | committed_mask)
            config_dependencies[slot] &= ~committed_mask
        if now >= next_issue:
            for offset in range(depth):
                slot = (head + offset) % depth
                if (
                    not valid[slot]
                    or issued[slot]
                    or allocated[slot] >= now
                    or uses[slot] & busy_mask
                ):
                    continue
                if dependencies[slot] or config_dependencies[slot]:
                    continue
                if read0[slot] >= alias_base and alias_busy[read0[slot] - alias_base]:
                    continue
                if read1[slot] >= alias_base and alias_busy[read1[slot] - alias_base]:
                    continue
                issued[slot] = True
                busy_mask |= uses[slot]
                count = 0
                if not config[slot]:
                    for v in range(alias_base):
                        if uses[slot] & (np.uint64(1) << np.uint64(v)):
                            assert groups[v] > 0
                            count += groups[v]
                physical_count[slot] = count
                busy_count += count
                f = function[slot]
                a = operand1[slot]
                b = operand2[slot]
                base = base_cycles[slot]
                if f == 16 or f == 33 or f == 49 or f == 69 or f == 70 or f == 72:
                    if not ((f == 16 or f == 33) and b >> np.uint64(63)):
                        v = (
                            np.int64(a & np.uint64(1023))
                            if f == 16 or f == 49 or f == 69 or f == 70 or f == 72
                            else np.int64((a >> np.uint64(20)) & np.uint64(1023))
                        )
                        base *= groups[v]
                if shape_model[f]:
                    rows = float(b & np.uint64(4095))
                    k = float(a >> np.uint64(30))
                    effective = 7.125 if rows == 1 else rows
                    duration = coefficients[f, 0] * (effective * k)
                    duration += (
                        coefficients[f, 1]
                        * float((b >> np.uint64(24)) & np.uint64(1))
                        * rows
                    )
                    duration += (
                        coefficients[f, 2]
                        * float((b >> np.uint64(25)) & np.uint64(1))
                        * rows
                    )
                    duration += coefficients[f, 3]
                else:
                    duration = base * scales[f]
                assert duration > 0
                completion[slot] = now + np.int64(math.ceil(duration))
                next_issue = now + 1
                break
        can_allocate = (
            index < n
            and not committed
            and live_entries < depth
            and now >= next_alloc
            and not alias_busy[tail]
            and alias_readers[alias_base + tail] == 0
        )
        if can_allocate and np.int64(commands[index]["arrival"]) <= now:
            slot = tail
            f = np.int64(commands[index]["f"])
            a = commands[index]["a"]
            b = commands[index]["b"]
            mode = f >> 4
            r0 = (
                np.int64(a & np.uint64(1023))
                if mode == 1
                or mode == 3
                or mode == 4
                or f == 32
                and b & np.uint64(4096)
                else -1
            )
            r1 = np.int64((a >> np.uint64(10)) & np.uint64(1023)) if mode == 4 else -1
            w = -1
            if (mode == 2 or mode == 3 or mode == 4) and f != 35:
                w = (
                    np.int64(a & np.uint64(1023))
                    if f == 32 and not b & np.uint64(4096)
                    else np.int64((a >> np.uint64(20)) & np.uint64(1023))
                )
            assert r0 < alias_base and r1 < alias_base and w < alias_base
            read_mask = np.uint64(0)
            if r0 >= 0:
                read_mask |= np.uint64(1) << np.uint64(r0)
            if r1 >= 0:
                read_mask |= np.uint64(1) << np.uint64(r1)
            write_mask = np.uint64(0) if w < 0 else np.uint64(1) << np.uint64(w)
            mask = read_mask | write_mask
            deps = np.uint64(0)
            conf_deps = np.uint64(0)
            for older in range(depth):
                if not valid[older]:
                    continue
                config_conflict = (
                    f == 32
                    and config[older]
                    or (f == 32 or config[older])
                    and bool(mask & uses[older])
                )
                conflict = bool(read_mask & writes[older] or write_mask & uses[older])
                same_ball = bids[f] >= 0 and bids[f] == bids[function[older]]
                if config_conflict:
                    conf_deps |= np.uint64(1) << np.uint64(older)
                elif not done[older] and (conflict or same_ball):
                    deps |= np.uint64(1) << np.uint64(older)
            valid[slot] = True
            issued[slot] = False
            done[slot] = False
            config[slot] = f == 32
            function[slot] = f
            operand1[slot] = a
            operand2[slot] = b
            base_cycles[slot] = commands[index]["base"]
            allocated[slot] = now
            physical_count[slot] = 0
            write_bank[slot] = w
            uses[slot] = mask
            writes[slot] = write_mask
            dependencies[slot] = deps
            config_dependencies[slot] = conf_deps
            read0[slot] = -1 if r0 < 0 else current_alias[r0]
            read1[slot] = -1 if r1 < 0 else current_alias[r1]
            if read0[slot] >= 0:
                alias_readers[read0[slot]] += 1
            if read1[slot] >= 0:
                alias_readers[read1[slot]] += 1
            if w >= 0:
                assert not alias_live[slot]
                current_alias[w] = alias_base + slot
                alias_live[slot] = True
                alias_busy[slot] = True
                alias_count += 1
            tail = (tail + 1) % depth
            live_entries += 1
            index += 1
            next_alloc = now + 1
            admitted = True
        else:
            admitted = False
        assert (
            busy_count >= 0
            and busy_count <= mapped_count
            and mapped_count <= capacity
            and alias_count <= depth
        )
        if (
            mapped_count != mapped[points - 1]
            or alias_count != aliases[points - 1]
            or busy_count != reserved[points - 1]
        ):
            times[points] = now
            mapped[points] = mapped_count
            aliases[points] = alias_count
            reserved[points] = busy_count
            points += 1
        if admitted:
            continue
        if index == n and live_entries == 0:
            break
        next_time = np.int64(9223372036854775807)
        if index < n and np.int64(commands[index]["arrival"]) > now:
            next_time = min(next_time, np.int64(commands[index]["arrival"]))
        for slot in range(depth):
            if not valid[slot]:
                continue
            if issued[slot] and not done[slot]:
                next_time = min(next_time, completion[slot])
            if (
                not issued[slot]
                and not dependencies[slot]
                and not config_dependencies[slot]
                and not uses[slot] & busy_mask
            ):
                hazard = (
                    read0[slot] >= alias_base
                    and alias_busy[read0[slot] - alias_base]
                    or read1[slot] >= alias_base
                    and alias_busy[read1[slot] - alias_base]
                )
                if not hazard:
                    next_time = min(next_time, max(now + 1, next_issue))
        if live_entries and done[head]:
            next_time = min(next_time, max(now + 1, completion[head] + 1))
        if committed:
            next_time = min(next_time, now + 1)
        if (
            index < n
            and live_entries < depth
            and not alias_busy[tail]
            and alias_readers[alias_base + tail] == 0
            and next_alloc > now
        ):
            next_time = min(next_time, next_alloc)
        assert next_time > now and next_time < 9223372036854775807
        now = next_time
    times[points] = now
    mapped[points] = mapped_count
    aliases[points] = alias_count
    reserved[points] = busy_count
    return (
        times[: points + 1],
        mapped[: points + 1],
        aliases[: points + 1],
        reserved[: points + 1],
        now,
    )


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
    capacity = thread.mem.bank.num
    depth = thread.frontend.rob_entries
    alias_base = core.virtual_bank_count + (2 if thread.gpdomain.enable else 0)
    assert capacity < 256 and depth < 64 and alias_base < 64
    calibration = json.loads(args.calibration.read_text())
    bids = np.full(128, -1, np.int64)
    scales = np.zeros(128)
    shape_model = np.zeros(128, np.bool_)
    coefficients = np.zeros((128, 4))
    for ball in thread.balldomain.balls:
        for isa in ball.instructions:
            bids[isa.funct7] = ball.id
    for f, row in calibration.items():
        if "model" in row:
            assert (
                row["model"]
                == "effective_rows*K + bias_rows + writeback_rows + startup"
            )
            shape_model[int(f)] = True
            coefficients[int(f)] = row["coefficients"]
        else:
            scales[int(f)] = row["scale"]
    dtype = np.dtype(
        [("f", "u1"), ("a", "u8"), ("b", "u8"), ("base", "u8"), ("arrival", "u8")]
    )
    started = time.monotonic()

    def instructions():
        first = None
        with (run_log / "bdb.ndjson").open() as stream:
            for line in stream:
                e = json.loads(line)
                if e["type"] != "itrace":
                    continue
                f = int(e["funct"], 16)
                assert f != 1, "Barrier replay requires a separate drain boundary"
                if first is None:
                    first = e["issue_cycle"]
                assert e["modeled_cycles"] is not None
                assert str(f) in calibration
                yield (
                    f,
                    int(e["rs1"], 16),
                    int(e["rs2"], 16),
                    e["modeled_cycles"],
                    e["issue_cycle"] - first,
                )

    commands = np.fromiter(instructions(), dtype=dtype)
    parsed = time.monotonic()
    t, m, a, b, end = simulate(
        commands, capacity, depth, alias_base, bids, scales, shape_model, coefficients
    )
    finished = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "bank-replay-events.npz",
        cycle=t,
        allocated_physical_banks=m,
        live_bat_aliases=a,
        busy_physical_banks=b,
    )
    result = {
        "method": (
            "Native event-driven replay of architectural-bank dependencies before the DMA issue- "
            "order fix, BAT alias lifetime and MSET physical mappings. Calibrated completion "
            "estimates; not cycle-accurate RTL or FPGA measurement."
        ),
        "limitations": [
            (
                "The four-slot thread admission interface is not modeled; ROB occupancy may exceed "
                "what the current issue path can supply."
            ),
            "The per-producer DMA issue-order dependency added after the MVOUT deadlock diagnosis is not modeled.",
            "Issue transport/endpoint backpressure is not modeled cycle by cycle.",
            "Physical mapping changes are approximated at MSET commit, rather than actual backend config handshake.",
            "Latency calibration extrapolates outside measured shapes.",
            "Alias count is not a physical-bank count.",
        ],
        "log_dir": str(run_log),
        "design_pb": str(args.design_pb),
        "thread_index": args.thread_index,
        "rob_entries": depth,
        "physical_banks": capacity,
        "bank_bytes": thread.mem.bank.width // 8 * thread.mem.bank.entries,
        "instructions": len(commands),
        "estimated_cycles": end,
        "offline_seconds": finished - started,
        "parse_seconds": parsed - started,
        "replay_and_jit_seconds": finished - parsed,
        "physical_allocation_peak": int(m.max()),
        "bat_alias_peak": int(a.max()),
        "busy_physical_peak": int(b.max()),
        "final_physical_banks": int(m[-1]),
        "final_bat_aliases": int(a[-1]),
    }
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
