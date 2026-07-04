import os
import re
import shlex
import sys
import tomllib

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.stream_run import StreamResult, stream_run_logger


def default_test_name(ball: str) -> str:
    return f"{ball}_ball_test"


def config_dir_name(config: str) -> str:
    if not isinstance(config, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", config):
        raise ValueError("invalid config name")
    return config


def uvm_paths(bbdir: str, input_data: dict, ball_override: str | None = None) -> dict:
    ball = ball_override or input_data.get("ball")
    if not isinstance(ball, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", ball):
        raise ValueError("invalid ball name")

    config = input_data.get("config")
    config_dir = config_dir_name(config) if config else None
    ball_dir = os.path.join(bbdir, "examples", "balls", ball)
    verify_dir = os.path.join(ball_dir, "verify")
    casegen_dir = os.path.join(verify_dir, "casegen")
    verify_env = os.path.join(bbdir, "verify")
    build_dir = os.path.join(verify_dir, "build")
    sim_dir = os.path.join(build_dir, config_dir) if config_dir else os.path.join(build_dir, "current")
    simv = os.path.join(sim_dir, "simv")
    csrc_dir = os.path.join(sim_dir, "csrc")
    rtl_dir = os.path.join(bbdir, "arch", "build", config_dir) if config_dir else None

    return {
        "ball": ball,
        "config": config_dir,
        "ball_dir": ball_dir,
        "verify_dir": verify_dir,
        "casegen_dir": casegen_dir,
        "verify_env": verify_env,
        "build_dir": build_dir,
        "sim_dir": sim_dir,
        "simv": simv,
        "csrc_dir": csrc_dir,
        "rtl_dir": rtl_dir,
    }


def checked_paths(bbdir: str, input_data: dict, ball_override: str | None = None) -> dict:
    paths = uvm_paths(bbdir, input_data, ball_override)
    for key in ("ball_dir", "verify_dir", "casegen_dir", "verify_env"):
        if not os.path.isdir(paths[key]):
            raise FileNotFoundError(f"{key} not found: {paths[key]}")
    if paths["rtl_dir"] and not os.path.isdir(paths["rtl_dir"]):
        raise FileNotFoundError(f"rtl_dir not found: {paths['rtl_dir']}")

    cargo_toml = os.path.join(paths["casegen_dir"], "Cargo.toml")
    if not os.path.isfile(cargo_toml):
        raise FileNotFoundError(f"Cargo.toml not found: {cargo_toml}")

    filelist = resolve_filelist(paths["verify_dir"], input_data.get("filelist"), paths["ball"])
    crate = read_crate_name(cargo_toml)
    dpi_lib = os.path.join(paths["casegen_dir"], "target", "debug", f"lib{crate.replace('-', '_')}")

    paths.update({
        "cargo_toml": cargo_toml,
        "filelist": filelist,
        "filelist_arg": prepare_filelist(paths, filelist),
        "crate": crate,
        "dpi_lib": dpi_lib,
    })
    return paths


def checked_run_paths(bbdir: str, input_data: dict) -> dict:
    paths = uvm_paths(bbdir, input_data)
    for key in ("ball_dir", "verify_dir", "casegen_dir", "verify_env"):
        if not os.path.isdir(paths[key]):
            raise FileNotFoundError(f"{key} not found: {paths[key]}")

    cargo_toml = os.path.join(paths["casegen_dir"], "Cargo.toml")
    if not os.path.isfile(cargo_toml):
        raise FileNotFoundError(f"Cargo.toml not found: {cargo_toml}")

    crate = read_crate_name(cargo_toml)
    dpi_lib = os.path.join(paths["casegen_dir"], "target", "debug", f"lib{crate.replace('-', '_')}")
    paths.update({
        "cargo_toml": cargo_toml,
        "crate": crate,
        "dpi_lib": dpi_lib,
    })
    return paths


def discover_uvm_balls(bbdir: str) -> list[str]:
    balls_dir = os.path.join(bbdir, "examples", "balls")
    if not os.path.isdir(balls_dir):
        return []
    balls = []
    for name in sorted(os.listdir(balls_dir)):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            continue
        verify_dir = os.path.join(balls_dir, name, "verify")
        filelist_dir = os.path.join(verify_dir, "filelists")
        casegen_toml = os.path.join(verify_dir, "casegen", "Cargo.toml")
        if os.path.isdir(filelist_dir) and os.path.isfile(casegen_toml):
            balls.append(name)
    return balls


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


def prepare_filelist(paths: dict, filelist: str) -> str:
    os.makedirs(paths["sim_dir"], exist_ok=True)
    generated = os.path.join(paths["sim_dir"], os.path.basename(filelist))
    rel_rtl_dir = os.path.relpath(paths["rtl_dir"], paths["verify_dir"]) if paths["rtl_dir"] else None
    rel_uvm_dir = os.path.relpath(os.path.join(paths["verify_env"], "uvm"), paths["verify_dir"])
    with open(filelist, "r") as src, open(generated, "w") as dst:
        for line in src:
            original = line
            line = line.replace("@UVM@", rel_uvm_dir)
            if "@RTL@" in line:
                if not rel_rtl_dir:
                    raise ValueError("filelist uses @RTL@ but --config was not provided")
                line = line.replace("@RTL@", rel_rtl_dir)
            elif re.search(r"(?:\.\./)+arch/build/[^/\s]+/", original):
                raise ValueError("filelist must use @RTL@ instead of hard-coded arch/build/<config>")
            if re.search(r"(?:\.\./)+verify/uvm/", original):
                raise ValueError("filelist must use @UVM@ instead of hard-coded verify/uvm")
            dst.write(line)
    return filelist_arg(paths["verify_dir"], generated)


def read_crate_name(cargo_toml: str) -> str:
    with open(cargo_toml, "rb") as f:
        data = tomllib.load(f)
    name = data.get("package", {}).get("name")
    if not name:
        raise ValueError(f"package.name not found in {cargo_toml}")
    return name


def failed_result(message: str) -> StreamResult:
    return StreamResult(returncode=1, stdout="", stderr=message)


def run_uvm_build_one(bbdir: str, input_data: dict, ctx, ball: str) -> tuple[StreamResult, dict]:
    try:
        paths = checked_paths(bbdir, input_data, ball)
    except Exception as e:
        ctx.logger.error(str(e))
        return failed_result(str(e)), {"task": "build", "error": str(e)}

    info = {
        "task": "build",
        "ball": paths["ball"],
        "config": paths["config"],
        "verify_dir": paths["verify_dir"],
        "filelist": paths["filelist"],
        "resolved_filelist": os.path.join(paths["verify_dir"], paths["filelist_arg"]),
        "simv": paths["simv"],
        "dpi_lib": paths["dpi_lib"],
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
        f"rm -rf {shlex.quote(paths['csrc_dir'])} {shlex.quote(paths['simv'])} {shlex.quote(paths['simv'])}.daidir && "
        f"mkdir -p {shlex.quote(paths['sim_dir'])} {shlex.quote(paths['csrc_dir'])} && "
        "vcs -full64 -sverilog -timescale=1ns/1ps -hsopt=off "
        "$VCS_UVM_ARGS "
        f"-Mdir={shlex.quote(paths['csrc_dir'])} "
        f"-o {shlex.quote(paths['simv'])} "
        f"-f {shlex.quote(paths['filelist_arg'])}"
    )
    vcs_cmd = f"nix develop {shlex.quote(paths['verify_env'])} --command bash -lc {shlex.quote(script)}"

    ctx.logger.info(f"Building UVM simulation for ball {paths['ball']} config {paths['config']}")
    vcs_result = stream_run_logger(
        cmd=vcs_cmd,
        logger=ctx.logger,
        cwd=bbdir,
        stdout_prefix="uvm vcs build",
        stderr_prefix="uvm vcs build",
    )
    if vcs_result.returncode == 0:
        current = os.path.join(paths["build_dir"], "current")
        os.makedirs(paths["build_dir"], exist_ok=True)
        tmp = f"{current}.tmp"
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        os.symlink(paths["config"], tmp)
        os.replace(tmp, current)
    return vcs_result, info


def run_uvm_build(bbdir: str, input_data: dict, ctx) -> tuple[StreamResult, dict]:
    config = input_data.get("config")
    if not config or config is True:
        return failed_result("Missing required parameter: --config=<name>"), {
            "task": "build",
            "error": "missing_config",
        }

    ball = input_data.get("ball")
    balls = [ball] if ball and ball is not True else discover_uvm_balls(bbdir)
    if not balls:
        return failed_result("No UVM balls found"), {"task": "build", "error": "no_uvm_balls"}

    built = []
    for one_ball in balls:
        result, info = run_uvm_build_one(bbdir, input_data, ctx, one_ball)
        if result.returncode != 0:
            return result, {**info, "built": built}
        built.append(one_ball)

    return StreamResult(returncode=0, stdout="", stderr=""), {
        "task": "build",
        "config": config,
        "built": built,
    }
