import re
import shlex
from pathlib import Path

from utils.stream_run import stream_run_logger
from steps.model.scripts.config import compile_config, cmake_arguments


def validate_model_build(repo: str, params: dict) -> list[str]:
    unknown = params.keys() - {"chip", "model", "ctrace", "dtrace"}
    if unknown:
        raise ValueError(f"Unknown model build parameters: {', '.join(sorted(unknown))}")
    for name in ("chip", "model"):
        value = params.get(name)
        if name == "model" and value is None:
            continue
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError(f"Invalid or missing {name}: {value}")
    recipes = Path(repo) / "examples/models" / params["chip"]
    model = params.get("model")
    if model is not None:
        if not (recipes / model / "CMakeLists.txt").is_file():
            raise ValueError(f"Model recipe does not exist: {recipes / model}")
        models = [model]
    else:
        models = sorted(path.parent.name for path in recipes.glob("*/CMakeLists.txt"))
        if not models:
            raise ValueError(f"No model recipes for chip: {params['chip']}")
    for name in ("ctrace", "dtrace"):
        if not isinstance(params.get(name, False), bool):
            raise ValueError(f"{name} must be a boolean flag")
    return models


def build_model(repo: str, chip: str, model: str | None = None, *, ctrace: bool = False, dtrace: bool = False,
                logger, task_scope: str) -> None:
    models = validate_model_build(repo, {"chip": chip, "model": model, "ctrace": ctrace, "dtrace": dtrace})
    source = Path(repo) / "stack/models"
    compiler = Path(repo) / "stack/compiler/thirdparty/buddy-mlir/build" / chip
    command = ["cmake", "--build", str(compiler), "--target", "buddy-opt",
               "buddy-translate", "buddy-llc", "rax-pack", "python-package-buddy", "--parallel", "4"]
    result = stream_run_logger(cmd=shlex.join(command), cwd=repo, logger=logger,
        stdout_prefix="model compiler", stderr_prefix="model compiler", task_scope=task_scope)
    if result.returncode:
        raise RuntimeError(f"Model compiler build failed ({result.returncode}): {shlex.join(command)}")
    for model in models:
        build = source / "build" / chip / model
        config = source / "models" / model / "configs/model.toml"
        model_args = []
        if config.is_file():
            pb = build / "config/model.pb"
            compile_config(config, pb, model)
            model_args = cmake_arguments(pb, model)
        commands = [
            ["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja",
             f"-DCHIP={chip}", f"-DMODEL={model}", f"-DCTRACE={'ON' if ctrace else 'OFF'}",
             f"-DDTRACE={'ON' if dtrace else 'OFF'}", *model_args],
            ["cmake", "--build", str(build), "--parallel", "1"],
        ]
        for command in commands:
            result = stream_run_logger(
                cmd=shlex.join(command), cwd=repo, logger=logger,
                stdout_prefix=f"model {model}", stderr_prefix=f"model {model}",
                task_scope=task_scope,
            )
            if result.returncode:
                raise RuntimeError(f"Model build failed ({result.returncode}): {shlex.join(command)}")
