import os
import re


def get_buckyball_path():
    current_dir = os.path.dirname(__file__)
    # bbdev/api/utils -> bbdev/api -> bbdev -> buckyball
    return os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))


def get_verilator_build_dir(bbdir, config=None, output_dir=None):
    if output_dir:
        return output_dir

    return get_config_build_dir(bbdir, config)


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


def get_dc_rtl_dir(bbdir, config=None, base_dir=None):
    if not config or config is True:
        raise ValueError("missing required parameter: config")

    name = sanitize_config_name(config)
    if not name:
        raise ValueError("invalid config name")

    if base_dir is True:
        raise ValueError("dir requires a path value")
    root = base_dir or os.path.join(bbdir, "arch", "build")
    if not os.path.isabs(root):
        raise ValueError("dir must be an absolute path")
    return os.path.join(root, name)


def check_dc_rtl_args(body: dict):
    allowed = {"config", "dir"}
    for name in body:
        if name not in allowed:
            raise ValueError(f"unexpected parameter: {name}")
