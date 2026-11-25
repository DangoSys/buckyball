"""Compare coherent DMA readback bytes with a frozen BEMU slice reference.

Both arguments name run directories under log/. The reference contains the
ELF, readback.json and BEMU instruction/memory traces. The subject contains
BEMU or RTL memory traces. A simulator exit code alone is not acceptance.
"""

import argparse
import hashlib
import json
import mmap
import struct
from pathlib import Path


def read_output(reference, subject):
    with (reference / "slice.elf").open("rb") as file, mmap.mmap(
        file.fileno(), 0, access=mmap.ACCESS_READ
    ) as data:
        header = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
        assert header[0][:6] == b"\x7fELF\x02\x01"
        sections = [
            struct.unpack_from("<IIQQQQIIQQ", data, header[6] + i * header[11])
            for i in range(header[12])
        ]
        names = sections[header[13]]
        strings = data[names[4] : names[4] + names[5]]
        record = next(
            s
            for s in sections
            if strings[s[0] :].split(b"\0", 1)[0] == b".gvn.bank_output"
        )
        address, expected, size = struct.unpack_from("<QQQ", data, record[4])
        section = next(
            s
            for s in sections
            if s[1] != 8 and s[3] <= expected and expected + size <= s[3] + s[5]
        )
        offset = section[4] + expected - section[3]
        expected_bytes = data[offset : offset + size]
    metadata = json.loads((reference / "readback.json").read_text())
    bank = metadata["bank"]
    ranges = metadata["ranges"]
    assert sum(last - first for first, last in ranges) == size
    assert all(0 <= first < last for first, last in ranges)

    # Match the verification MVINs by their source interval and architectural
    # bank. Their instruction IDs also identify the writes in the RTL trace.
    commands = []
    reference_writes = []
    for path in reference.rglob("bdb.ndjson"):
        if "golden" in path.relative_to(reference).parts:
            continue
        rows, instructions = {}, []
        for line in path.open():
            event = json.loads(line)
            if (
                event["type"] == "mtrace"
                and event["event"] == "write"
                and event["vbank_id"] == bank
            ):
                rows.setdefault(event["rob_id"], []).append(event)
                reference_writes.append(event)
            if event["type"] != "itrace" or event["funct"] != "0x21":
                continue
            rs1, rs2 = int(event["rs1"], 16), int(event["rs2"], 16)
            source = rs2 & ((1 << 39) - 1)
            if (rs1 >> 20) & 511 == bank and any(
                address + first <= source < address + last for first, last in ranges
            ):
                instructions.append(
                    (event["event_index"], source, (rs1 >> 30) & ((1 << 33) - 1))
                )
        for inst, source, count in instructions:
            commands.append(
                {
                    "hart": rows[inst][0]["hart_id"],
                    "inst": inst,
                    "address": source,
                    "rows": count,
                }
            )
    commands.sort(key=lambda command: command["address"])
    remaining = iter(commands)
    for first, last in ranges:
        cursor = address + first
        while cursor < address + last:
            command = next(remaining)
            assert command["address"] == cursor, "Readback has a gap or overlap"
            command["bytes"] = min(command["rows"] * 16, address + last - cursor)
            cursor += command["rows"] * 16
        assert (
            address + last <= cursor < address + last + 16
        ), "Invalid readback padding"
    assert next(remaining, None) is None, "Unexpected readback instructions"

    events = reference_writes if reference == subject else []
    if reference != subject:
        for path in subject.rglob("bdb.ndjson"):
            if "golden" in path.relative_to(subject).parts:
                continue
            for line in path.open():
                event = json.loads(line)
                if event["type"] in ("bankmap", "mtrace"):
                    events.append(event)
    events.sort(
        key=lambda event: event["clk"] if "clk" in event else event["event_index"]
    )
    mapping, writes = {}, {}
    wanted = {(command["hart"], command["inst"]) for command in commands}
    for event in events:
        hart, version = event["hart_id"], event["vbank_id"]
        if event["type"] == "bankmap":
            if event["event"] == "bind":
                mapping[hart, version] = event["bank_id"]
            continue
        if event["event"] != "write":
            continue
        rtl = "inst_id" in event
        architectural = mapping[hart, version] if rtl else version
        inst = event["inst_id"] if rtl else event["rob_id"]
        if architectural == bank and (hart, inst) in wanted:
            writes.setdefault((hart, inst), []).append(event)

    actual = bytearray()
    for command in commands:
        value = bytearray(command["rows"] * 16)
        seen = bytearray(len(value))
        for event in writes[command["hart"], command["inst"]]:
            assert event["group_id"] == 0
            row = int(event["addr"], 16)
            word = int(event["data"], 16).to_bytes(16, "little")
            mask = int(event["write_mask"], 16)
            for byte in range(16):
                if mask & (1 << byte):
                    value[row * 16 + byte] = word[byte]
                    seen[row * 16 + byte] = 1
        assert all(seen), "Readback trace is missing bytes"
        actual.extend(value[: command["bytes"]])
    result = {
        "bytes": size,
        "bank": bank,
        "commands": commands,
        "reference": str(reference),
        "subject": str(subject),
    }
    return bytes(actual), expected_bytes, offset, result


def check(reference, subject):
    actual, expected_bytes, _, result = read_output(reference, subject)
    if actual != expected_bytes:
        first = next(
            i
            for i, pair in enumerate(zip(actual, expected_bytes))
            if pair[0] != pair[1]
        )
        raise ValueError(
            f"Coherent output differs at byte {first}: "
            f"actual {actual[first]}, expected {expected_bytes[first]}"
        )
    result["passed"] = True
    (subject / "bank-output-check.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def capture(reference):
    """Freeze a BEMU readback without printing result words on the guest CPU."""
    actual, _, offset, result = read_output(reference, reference)
    data = bytearray((reference / "slice.elf").read_bytes())
    result["input_sha256"] = hashlib.sha256(data).hexdigest()
    (reference / "capture.elf").write_bytes(data)
    data[offset : offset + len(actual)] = actual
    (reference / "slice.elf").write_bytes(data)
    (reference / "reference.elf").write_bytes(data)
    result["reference_sha256"] = hashlib.sha256(data).hexdigest()
    result["captured"] = True
    (reference / "bank-output-capture.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("subject")
    args = parser.parse_args()
    logs = Path(__file__).resolve().parent.parent / "log"
    for name in (args.reference, args.subject):
        if Path(name).name != name:
            parser.error("Use a run directory name under log/")
    result = check(logs / args.reference, logs / args.subject)
    print(f"PASS: exact coherent bank output, {result['bytes']} bytes")
