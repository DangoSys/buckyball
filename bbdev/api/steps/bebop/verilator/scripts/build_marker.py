import json
import os


def build_marker_path(bebop_dir: str) -> str:
  return os.path.join(bebop_dir, "target", "debug", ".bbdev-verilator-build.json")


def write_build_marker(bebop_dir: str, config: str, vsrc_dir: str, binary: str):
  marker = build_marker_path(bebop_dir)
  os.makedirs(os.path.dirname(marker), exist_ok=True)
  data = {
    "config": config,
    "vsrc_dir": os.path.abspath(vsrc_dir),
    "binary": os.path.abspath(binary),
  }
  tmp = f"{marker}.tmp"
  with open(tmp, "w") as f:
    json.dump(data, f, indent=2, sort_keys=True)
    f.write("\n")
  os.replace(tmp, marker)


def read_build_marker(bebop_dir: str) -> dict:
  marker = build_marker_path(bebop_dir)
  with open(marker) as f:
    return json.load(f)
