import os
import re


def get_buckyball_path():
    current_dir = os.path.dirname(__file__)
    # bbdev/api/utils -> bbdev/api -> bbdev -> buckyball
    return os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))


def get_verilator_build_dir(bbdir, config=None, output_dir=None):
    if output_dir:
        return output_dir

    if config and config != "None":
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", config).strip("._")
        if name:
            return f"{bbdir}/arch/build/{name}"

    return f"{bbdir}/arch/build"
