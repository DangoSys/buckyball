"""Separate firtool's concatenated ancillary file list from its Verilog.

Every RTL byte is retained. The original Chisel output is saved for auditing;
the only excluded section is firrtl_black_box_resource_files.f (not RTL).
"""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
target = root / "generated/PebbleA7Chip.sv"
marker = b'// ----- 8< ----- FILE "firrtl_black_box_resource_files.f" ----- 8< -----'
raw = target.read_bytes()
assert raw.count(marker) == 1, "Expected exactly one ancillary .f section"
rtl, filelist = raw.split(marker)
assert all(line.endswith(b".v") for line in filelist.splitlines() if line.strip())
target.with_suffix(".emitted.txt").write_bytes(raw)
target.write_bytes(rtl)
target.with_name("resource-files.f").write_bytes(filelist)
record = {
    "operation": "separate trailing ancillary .f list; no RTL edits",
    "emitted_sha256": hashlib.sha256(raw).hexdigest(),
    "rtl_sha256": hashlib.sha256(rtl).hexdigest(),
}
target.with_name("RTL_MANIFEST.json").write_text(json.dumps(record, indent=2) + "\n")
print(record)
