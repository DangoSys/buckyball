import json
import os


def build_marker_path(target_dir: str) -> str:
    return os.path.join(target_dir, ".bbdev-verilator-build.json")


def write_build_marker(
    target_dir: str,
    config: str,
    vsrc_dir: str,
    binary: str,
    diff: bool = False,
):
    marker = build_marker_path(target_dir)
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    data = {
        "config": config,
        "vsrc_dir": os.path.abspath(vsrc_dir),
        "binary": os.path.abspath(binary),
        "diff": diff,
    }
    tmp = f"{marker}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, marker)


def read_build_marker(target_dir: str) -> dict:
    marker = build_marker_path(target_dir)
    with open(marker) as f:
        return json.load(f)
