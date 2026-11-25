"""
Common utility functions for all event steps.
"""

import importlib.util
import re

from pathlib import Path

import iii.iii

from utils.path import get_buckyball_path
from utils.process_registry import set_current_task_scope

# The local iii engine answers state and queue calls slowly when the host is
# loaded and its disk busy. The client's 30 s default failed hour-long runs on
# one slow write (a step's final state never recorded, its waiter waiting on);
# a call that never returns still fails, after 10 minutes.
iii.iii.III.call.__defaults__ = (600.0,)


def require_target(data: dict, kinds=("device", "hart")) -> str:
    """The selection of a request, <level>/<name>/<config name>, with its
    design.pb installed: `--config <design>/.../<Name>.toml` is a config TOML
    Top under examples/designs, and exactly one of --device, --hart, --thread,
    --ball names the component of it to simulate. The design, the first
    directory of the config, is written to data["design"]. --device without
    --config is the device from its design's one <Design>DefaultConfig.toml,
    written to data["config"]. --ball without
    --config is the Ball alone from its default config: a name is a prefab
    Ball, a path <design>/balls/<ball> a design's; data["design"] is then the
    Ball's directory relative to examples. --thread and --hart without --config
    are the same for a thread or a hart."""
    selected = [kind for kind in ("device", "hart", "thread", "ball") if kind in data]
    if len(selected) != 1:
        raise ValueError(
            "Specify exactly one of --device <device>, --hart <hart>,"
            " --thread <thread>, --ball <ball>, and --config <config TOML> for a"
            " config other than the default"
        )
    kind = selected[0]
    if kind not in kinds:
        raise ValueError(f"--{kind} is not supported by this operation")
    if not isinstance(data[kind], str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+(/[A-Za-z0-9_-]+)*", data[kind]
    ):
        raise ValueError(f"--{kind} requires a name")
    config = data.get("config")
    if kind == "device" and config is None:
        # A device alone, from its design's one default config.
        repo = Path(get_buckyball_path())
        found = sorted(
            (repo / "examples/designs").glob(
                f"*/devices/{data[kind]}/configs/*DefaultConfig.toml"
            )
        )
        if len(found) != 1:
            raise ValueError(
                f"--device {data[kind]} needs exactly one"
                f" examples/designs/*/devices/{data[kind]}/configs/*DefaultConfig.toml,"
                f" found {len(found)}"
            )
        config = str(found[0].relative_to(repo / "examples/designs"))
        data["config"] = config
    if kind in ("ball", "thread", "hart") and config is None:
        # A Ball, a thread or a hart alone, from its own default config.
        repo = Path(get_buckyball_path())
        spec = importlib.util.spec_from_file_location(
            "config_install", repo / "bbdev/api/steps/config/scripts/install.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        selection = module.install(repo, "", kind, data[kind])
        root = {
            "ball": module.ball_root,
            "thread": module.thread_root,
            "hart": module.hart_root,
        }[kind](repo, data[kind])
        data["design"] = str(root.relative_to(repo / "examples"))
        return selection
    if "/" in data[kind]:
        raise ValueError(f"--{kind} with --config requires a name")
    if not isinstance(config, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+(/[A-Za-z0-9_-][A-Za-z0-9_.-]*)*\.toml", config
    ):
        raise ValueError("--config requires a TOML path relative to examples/designs")
    repo = Path(get_buckyball_path())
    spec = importlib.util.spec_from_file_location(
        "config_install", repo / "bbdev/api/steps/config/scripts/install.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selection = module.install(repo, config, kind, data[kind])
    data["design"] = Path(config).parts[0]
    return selection


def require_design(data: dict) -> str:
    """The design of a request, set by require_target."""
    return data["design"]


def require_product(data: dict) -> str:
    """The workload a simulation runs, by the name of what it loads into the
    DDR: `--model <model> --os (baremetal | linux)` is the model's
    <design>-<model>-<os>, `--bench <bench> --os (baremetal | linux)` a
    benchmark's <design>-<bench>-<os> (embench, spec), `--test <name>` a test
    by its whole name (a baremetal test, or the Linux regression image
    <design>-test-linux). Call after require_target, which sets
    data["design"]."""
    given = [kind for kind in ("model", "bench", "test") if kind in data]
    if len(given) != 1:
        raise ValueError(
            "Specify exactly one of --model <model> --os (baremetal | linux),"
            " --bench <bench> --os (baremetal | linux), --test <name>"
        )
    (kind,) = given
    if kind == "test":
        if "os" in data:
            raise ValueError(
                "--os goes with --model or --bench; a test's name carries its OS"
            )
        return data["test"]
    if data.get("os") not in ("baremetal", "linux"):
        raise ValueError(f"--{kind} needs --os (baremetal | linux)")
    return f"{data['design']}-{data[kind]}-{data['os']}"


def get_origin_trace_id(input_data, ctx):
    """Get the origin trace_id from input_data (passed by API step) or fall back to ctx.trace_id.

    iii 0.7+ assigns a new trace_id per handler invocation, so the API step's
    trace_id must be forwarded explicitly via input_data["_trace_id"].
    """
    trace_id = (
        input_data["_trace_id"]
        if isinstance(input_data, dict) and "_trace_id" in input_data
        else ctx.trace_id
    )
    set_current_task_scope(trace_id)
    return trace_id


async def check_result(
    ctx, returncode, continue_run=False, extra_fields=None, trace_id=None
):
    """
    Check returncode, create appropriate result objects and set state.

    Args:
        ctx: The flow context object
        returncode: The return code (int)
        continue_run: If True, set processing state instead of success/failure
        extra_fields: Optional dictionary of extra fields to include in result body
        trace_id: Optional trace_id to use as state scope (defaults to ctx.trace_id)

    Returns:
        tuple: (success_result, failure_result) - one will be None based on returncode and continue_run
    """
    extra_fields = extra_fields or {}
    scope = trace_id or ctx.trace_id

    if returncode != 0:
        failure_result = {
            "status": 500,
            "body": {
                "success": False,
                "failure": True,
                "processing": False,
                "returncode": returncode,
                **extra_fields,
            },
        }
        await ctx.state.set(scope, "failure", failure_result)
        return None, failure_result
    elif continue_run:
        await ctx.state.set(scope, "processing", {"processing": True, **extra_fields})
        return None, None
    else:
        success_result = {
            "status": 200,
            "body": {
                "success": True,
                "failure": False,
                "processing": False,
                "returncode": returncode,
                **extra_fields,
            },
        }
        await ctx.state.set(scope, "success", success_result)
        return success_result, None
