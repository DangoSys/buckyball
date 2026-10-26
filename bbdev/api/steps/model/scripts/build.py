import shlex
from pathlib import Path

from utils.stream_run import stream_run_logger


def build_model(
    repo: str, design: str, model: str | None = None, *, logger, task_scope: str
) -> None:
    """Build one model recipe of `design`, or every recipe of it when `model` is None.
    Models run on the whole device, so they build against its device entry."""
    recipes = Path(repo) / "examples/models" / design
    models = (
        [model]
        if model
        else sorted(path.parent.name for path in recipes.glob("*/CMakeLists.txt"))
    )
    source = Path(repo) / "stack/models"
    from steps.stack.scripts.build import build_design

    # Models compile with the design's compiler plugins and link its device runtime.
    build_design(
        repo, design=design, selection="device", logger=logger, task_scope=task_scope
    )
    for model in models:
        build = source / "build" / design / model
        build_command = ["cmake", "--build", str(build), "--parallel", "1"]
        build_command.extend(["--target", f"{model}-model-build"])
        # A recipe reads the prepared model while it configures, so the model
        # is prepared first, by its own project: its weights and checkpoints
        # are rebuilt only when their inputs change.
        prepare = source / "build" / "models" / model
        commands = [
            # Prepared weights can predate this build tree; use file timestamps.
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(prepare),
                "-G",
                "Unix Makefiles",
                f"-DMODEL={model}",
            ],
            ["cmake", "--build", str(prepare), "--target", f"{model}-prepare"],
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build),
                "-G",
                "Ninja",
                "-DARCH=buckyball",
                f"-DDESIGN={design}",
                "-DBUCKYBALL_SIM_TARGET=device",
                f"-DMODEL={model}",
            ],
            build_command,
        ]
        for command in commands:
            result = stream_run_logger(
                cmd=shlex.join(command),
                cwd=repo,
                logger=logger,
                stdout_prefix=f"model {model}",
                stderr_prefix=f"model {model}",
                task_scope=task_scope,
            )
            if result.returncode:
                raise RuntimeError(
                    f"Model build failed ({result.returncode}): {shlex.join(command)}"
                )
