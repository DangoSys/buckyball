"""Read-only audit of original Buckyball blobs and the generated RTL package."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "UPSTREAM_MANIFEST.json").read_text())
for name, digest in manifest["files"].items():
    assert (
        hashlib.sha256((root / "upstream" / name).read_bytes()).hexdigest() == digest
    ), name
generated = json.loads((root / "generated/RTL_MANIFEST.json").read_text())
for name, key in [
    ("PebbleA7Chip.sv", "rtl_sha256"),
    ("PebbleA7Chip.emitted.txt", "emitted_sha256"),
]:
    assert (
        hashlib.sha256((root / "generated" / name).read_bytes()).hexdigest()
        == generated[key]
    ), name
print(f"PASS: {len(manifest['files'])} original files and both Chisel output hashes")
