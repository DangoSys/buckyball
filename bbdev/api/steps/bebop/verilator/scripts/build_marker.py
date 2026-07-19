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


def read_build_marker(bebop_dir: str) -> dict | None:
  marker = build_marker_path(bebop_dir)
  try:
    with open(marker) as f:
      return json.load(f)
  except FileNotFoundError:
    return None


def verify_build_marker(bebop_dir: str, arch_config: str, vsrc_dir: str) -> dict | None:
  bebop_bin = os.path.join(bebop_dir, "target", "debug", "bebop")
  if not os.path.isfile(bebop_bin):
    return {"error": "bebop_binary_not_found", "binary": bebop_bin}

  try:
    marker = read_build_marker(bebop_dir)
  except (OSError, json.JSONDecodeError) as e:
    marker_path = build_marker_path(bebop_dir)
    return {"error": "build_marker_read_failed", "marker": marker_path, "detail": str(e)}

  expect_vsrc = os.path.abspath(vsrc_dir)
  expect_bin = os.path.abspath(bebop_bin)
  if marker is None:
    return {"error": "build_marker_not_found", "marker": build_marker_path(bebop_dir)}
  if (
    marker.get("config") != arch_config
    or marker.get("vsrc_dir") != expect_vsrc
    or marker.get("binary") != expect_bin
  ):
    return {
      "error": "build_marker_mismatch",
      "marker": marker,
      "expected": {
        "config": arch_config,
        "vsrc_dir": expect_vsrc,
        "binary": expect_bin,
      },
    }
  return None
