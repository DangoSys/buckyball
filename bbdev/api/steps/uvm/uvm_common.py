import os
import re
import shlex
import sys
import tomllib

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.stream_run import StreamResult, stream_run_logger


def default_test_name(ball: str) -> str:
    return f"{ball}_ball_test"


def uvm_paths(bbdir: str, input_data: dict) -> dict:
    ball = input_data.get("ball")
    if not isinstance(ball, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", ball):
        raise ValueError("invalid ball name")

    ball_dir = os.path.join(bbdir, "examples", "balls", ball)
    verify_dir = os.path.join(ball_dir, "verify")
    casegen_dir = os.path.join(verify_dir, "casegen")
    verify_env = os.path.join(bbdir, "verify")
    sim_dir = os.path.join(verify_dir, "build")
    simv = os.path.join(sim_dir, "simv")
    csrc_dir = os.path.join(sim_dir, "csrc")

    return {
        "ball": ball,
        "ball_dir": ball_dir,
        "verify_dir": verify_dir,
        "casegen_dir": casegen_dir,
        "verify_env": verify_env,
        "sim_dir": sim_dir,
        "simv": simv,
        "csrc_dir": csrc_dir,
    }


def checked_paths(bbdir: str, input_data: dict) -> dict:
    paths = uvm_paths(bbdir, input_data)
    for key in ("ball_dir", "verify_dir", "casegen_dir", "verify_env"):
        if not os.path.isdir(paths[key]):
            raise FileNotFoundError(f"{key} not found: {paths[key]}")

    cargo_toml = os.path.join(paths["casegen_dir"], "Cargo.toml")
    if not os.path.isfile(cargo_toml):
        raise FileNotFoundError(f"Cargo.toml not found: {cargo_toml}")

    filelist = resolve_filelist(paths["verify_dir"], input_data.get("filelist"), paths["ball"])
    crate = read_crate_name(cargo_toml)
    dpi_lib = os.path.join(paths["casegen_dir"], "target", "debug", f"lib{crate.replace('-', '_')}")

    paths.update({
        "cargo_toml": cargo_toml,
        "filelist": filelist,
        "filelist_arg": filelist_arg(paths["verify_dir"], filelist),
        "crate": crate,
        "dpi_lib": dpi_lib,
    })
    return paths


def resolve_filelist(verify_dir: str, arg, ball: str) -> str:
    if arg is True:
        raise ValueError("parameter --filelist requires a path value")

    if arg:
        path = arg if os.path.isabs(arg) else os.path.join(verify_dir, arg)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"filelist not found: {path}")
        return os.path.abspath(path)

    default = os.path.join(verify_dir, "filelists", f"{ball}_ball_toy.f")
    if os.path.isfile(default):
        return default

    filelist_dir = os.path.join(verify_dir, "filelists")
    if not os.path.isdir(filelist_dir):
        raise FileNotFoundError(f"filelist directory not found: {filelist_dir}")
    found = [
        os.path.join(filelist_dir, name)
        for name in os.listdir(filelist_dir)
        if name.endswith(".f")
    ]
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one filelist under {filelist_dir}, found {len(found)}")
    return found[0]


def filelist_arg(verify_dir: str, filelist: str) -> str:
    rel = os.path.relpath(filelist, verify_dir)
    if rel.startswith(".."):
        return filelist
    return rel


def read_crate_name(cargo_toml: str) -> str:
    with open(cargo_toml, "rb") as f:
        data = tomllib.load(f)
    name = data.get("package", {}).get("name")
    if not name:
        raise ValueError(f"package.name not found in {cargo_toml}")
    return name


def failed_result(message: str) -> StreamResult:
    return StreamResult(returncode=1, stdout="", stderr=message)


def run_uvm_build(bbdir: str, input_data: dict, ctx) -> tuple[StreamResult, dict]:
    try:
        paths = checked_paths(bbdir, input_data)
    except Exception as e:
        ctx.logger.error(str(e))
        return failed_result(str(e)), {"task": "build", "error": str(e)}

    info = {
        "task": "build",
        "ball": paths["ball"],
        "verify_dir": paths["verify_dir"],
        "filelist": paths["filelist"],
        "simv": paths["simv"],
    }

    cargo_cmd = (
        f"nix develop {shlex.quote(paths['verify_env'])} --command "
        f"cargo build --manifest-path {shlex.quote(paths['cargo_toml'])}"
    )
    ctx.logger.info(f"Building UVM DPI reference for ball {paths['ball']}")
    cargo_result = stream_run_logger(
        cmd=cargo_cmd,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="uvm dpi build",
        stderr_prefix="uvm dpi build",
    )
    if cargo_result.returncode != 0:
        return cargo_result, info

    script = (
        f"cd {shlex.quote(paths['verify_dir'])} && "
        f"mkdir -p {shlex.quote(paths['sim_dir'])} {shlex.quote(paths['csrc_dir'])} && "
        "vcs -full64 -sverilog -timescale=1ns/1ps "
        "$VCS_UVM_ARGS "
        f"-sv_lib {shlex.quote(paths['dpi_lib'])} "
        f"-Mdir {shlex.quote(paths['csrc_dir'])} "
        f"-o {shlex.quote(paths['simv'])} "
        f"-f {shlex.quote(paths['filelist_arg'])}"
    )
    vcs_cmd = f"nix develop {shlex.quote(paths['verify_env'])} --command zsh -lc {shlex.quote(script)}"

    ctx.logger.info(f"Building UVM simulation for ball {paths['ball']}")
    return stream_run_logger(
        cmd=vcs_cmd,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="uvm vcs build",
        stderr_prefix="uvm vcs build",
    ), info
