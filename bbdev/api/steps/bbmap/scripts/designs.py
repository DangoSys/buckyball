"""The design map: every design under examples/designs, what it is for (the
first paragraph of its README), the model recipes written for it
(stack/models/archs/**/<design>/<model>/), the benchmarks its device lists
(sims/device/regression/bemu/bench.toml; none without one), its one default
config (<Design>DefaultConfig) and the composition of that config, read from
the design.pb it installs.
Usage: designs.py <repo> <out.json>"""

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])
scripts = repo / "bbdev/api/steps/config/scripts"
sys.path.insert(0, str(scripts))
import design_pb2  # noqa: E402

designs_root = repo / "examples/designs"
archs = repo / "stack/models/archs"


def description(root: Path) -> str:
    """The first paragraph of the design's README, after its heading."""
    paragraphs = [p.strip() for p in (root / "README.md").read_text().split("\n\n")]
    return " ".join(paragraphs[1].split())


def models(design: str) -> list[str]:
    """The model recipes written for the design: the add_subdirectory entries
    of stack/models/archs/**/<design>/CMakeLists.txt."""
    return sorted(
        name
        for owner in archs.rglob(design)
        if (owner / "CMakeLists.txt").is_file()
        for name in re.findall(
            r"add_subdirectory\(([A-Za-z0-9_.-]+)\)",
            (owner / "CMakeLists.txt").read_text(),
        )
    )


def composition(design: str, config: str) -> dict:
    selection = subprocess.check_output(
        [
            sys.executable,
            str(scripts / "install.py"),
            str(repo),
            config,
            "device",
            design,
        ],
        text=True,
    ).strip()
    pb = design_pb2.Design()
    pb.ParseFromString(
        (
            designs_root / design / "output/sims" / selection / "generated/design.pb"
        ).read_bytes()
    )
    verified_ips = {p.ip for p in pb.verification.profiles}
    return {
        "ips": [
            {"ip": ip, "has_suite": ip in verified_ips} for ip in pb.verification.ips
        ],
        "harts": [
            {
                "id": h.hart_id,
                "component": h.component,
                "threads": len(h.thread_indices),
                # Balls on the hart's own accelerator, outside any thread.
                "balls": [b.component for b in h.accelerator.balldomain.balls],
            }
            for h in pb.device.harts
        ],
        "threads": [
            {"cpu": t.cpu.kind, "balls": [b.component for b in t.balldomain.balls]}
            for t in pb.threads
        ],
    }


entries = []
for root in sorted(
    p for p in designs_root.iterdir() if (p / "devices" / p.name).is_dir()
):
    design = root.name
    device = root / "devices" / design
    (default,) = (device / "configs").glob("*DefaultConfig.toml")
    config = str(default.relative_to(designs_root))
    listed = device / "sims/device/regression/bemu/bench.toml"
    entry = {
        "design": design,
        "description": description(root),
        "models": models(design),
        "benches": (
            tomllib.loads(listed.read_text())["bench"]["benches"]
            if listed.is_file()
            else []
        ),
        "config": config,
        "overview": composition(design, config),
    }
    entries.append(entry)

out.write_text(json.dumps({"designs": entries}, indent=2, sort_keys=True) + "\n")
print(f"{len(entries)} designs mapped into {out}")
