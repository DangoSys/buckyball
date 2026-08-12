import os
import re
from datetime import datetime
from pathlib import Path


def get_buckyball_path():
    root = os.environ.get("BUCKYBALL_ROOT")
    if root:
        if not os.path.isabs(root):
            raise ValueError("BUCKYBALL_ROOT must be an absolute path")
        if not os.path.isdir(root):
            raise ValueError(f"BUCKYBALL_ROOT does not exist: {root}")
        return root

    current_dir = os.path.dirname(__file__)
    # bbdev/api/utils -> bbdev/api -> bbdev -> buckyball
    return os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))


def get_verilator_build_dir(bbdir, config=None, output_dir=None):
    if output_dir:
        return output_dir

    return get_config_build_dir(bbdir, config)


def get_vcs_build_dir(bbdir, config=None, output_dir=None):
    """Return the shared generated-RTL directory used by the VCS flow.

    VCS and Verilator elaborate the same BBSimHarness RTL.  Simulator-specific
    products live below ``vcs/`` so neither flow overwrites the other.
    """
    return get_config_build_dir(bbdir, config, output_dir)


def get_chip_from_config(bbdir, config):
    """Resolve the owning chip from a fully qualified Scala config name."""
    if not isinstance(config, str) or not config or config == "None":
        raise ValueError("invalid config name")

    config_class = config.rsplit(".", 1)[-1]
    declaration = re.compile(rf"\b(?:class|object)\s+{re.escape(config_class)}\b")
    chips_root = Path(bbdir) / "examples" / "chips"
    matches = []
    for chip_dir in sorted(path for path in chips_root.iterdir() if path.is_dir()):
        scala_root = chip_dir / "arch" / "src" / "main" / "scala"
        if not scala_root.is_dir():
            continue
        for source in scala_root.rglob("*.scala"):
            try:
                contents = source.read_text(encoding="utf-8")
            except OSError:
                continue
            if declaration.search(contents):
                matches.append(chip_dir.name)
                break

    if len(matches) != 1:
        raise ValueError(
            f"config {config} must resolve to exactly one chip; matches={matches}"
        )
    return matches[0]


def sanitize_config_name(config=None):
    if config and config != "None":
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", config).strip("._")
        if name:
            return name
    return None


def get_config_build_dir(bbdir, config=None, output_dir=None, output_root=None):
    if output_dir:
        return output_dir

    name = sanitize_config_name(config)
    if output_root:
        if not name:
            raise ValueError("output_root requires a valid config name")
        return os.path.join(output_root, name)

    if name:
        return f"{bbdir}/arch/build/{name}"

    return f"{bbdir}/arch/build"


def get_dc_rtl_dir(bbdir, config=None):
    if not config or config is True:
        raise ValueError("missing required parameter: config")

    name = sanitize_config_name(config)
    if not name:
        raise ValueError("invalid config name")

    return os.path.join(bbdir, "arch", "build", name)


def get_dc_analysis_dir(bbdir, config=None, stage="area"):
    if not config or config is True:
        raise ValueError("missing required parameter: config")
    name = sanitize_config_name(config)
    if not name:
        raise ValueError("invalid config name")
    if stage not in {"area", "power"}:
        raise ValueError(f"invalid DC analysis stage: {stage}")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return os.path.join(bbdir, "log", f"{timestamp}-dc-{name}", stage)


def check_dc_rtl_args(body: dict):
    allowed = {"config", "top"}
    for name in body:
        if name not in allowed:
            raise ValueError(f"unexpected parameter: {name}")
    top = body.get("top")
    if top is not None and (not isinstance(top, str) or not top):
        raise ValueError("top must be a non-empty module name")
    if isinstance(top, str) and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", top):
        raise ValueError("top must be a legal unescaped Verilog module name")


def check_dc_power_args(body: dict):
    allowed = {"config", "top", "activity", "format", "strip_path", "workload", "start_ns", "end_ns", "start-ns", "end-ns"}
    for name in body:
        if name not in allowed:
            raise ValueError(f"unexpected parameter: {name}")
    common = {key: body[key] for key in ("config", "top") if key in body}
    check_dc_rtl_args(common)
    activity = body.get("activity")
    activity_format = body.get("format")
    if activity is not None and (not isinstance(activity, str) or not activity):
        raise ValueError("activity must be a non-empty path when provided")
    if activity_format is not None and activity_format not in {"saif", "vcd", "fsdb"}:
        raise ValueError("format must be one of saif, vcd, or fsdb")
    if (activity is None) != (activity_format is None):
        raise ValueError("activity and format must be provided together")
    strip_path = body.get("strip_path")
    if strip_path is not None and not isinstance(strip_path, str):
        raise ValueError("strip_path must be a string")
    for name in ("workload",):
        value = body.get(name)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{name} must be a non-empty path when provided")
        if isinstance(value, str) and os.path.isabs(value):
            raise ValueError(f"{name} must be a built ELF name, not an absolute path")
    for name in ("start_ns", "end_ns", "start-ns", "end-ns"):
        value = body.get(name)
        if value is not None:
            try:
                if float(value) < 0:
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a non-negative number") from exc
    start = body.get("start_ns", body.get("start-ns"))
    end = body.get("end_ns", body.get("end-ns"))
    if start is not None and end is not None and float(start) >= float(end):
        raise ValueError("start_ns must be smaller than end_ns")
