import re
import shlex
from pathlib import Path

from utils.stream_run import stream_run_logger


def build_model(
    repo: str,
    design: str,
    model: str,
    *,
    selection: str,
    logger,
    task_scope: str,
) -> None:
    """Build one model recipe of `design`, or every recipe of it when `model` is "all".
    Models run on the whole device, so they build against a device config. The
    design's recipes sit under stack/models/archs/<arch>/.../<design>/, listed
    by its CMakeLists.txt; <arch> selects the recipe tree."""
    source = Path(repo) / "stack/models"
    archs = source / "archs"
    (owner,) = [p for p in archs.rglob(design) if (p / "CMakeLists.txt").is_file()]
    arch = owner.relative_to(archs).parts[0]
    models = (
        re.findall(
            r"add_subdirectory\(([A-Za-z0-9_.-]+)\)",
            (owner / "CMakeLists.txt").read_text(),
        )
        if model == "all"
        else [model]
    )
    environment = [
        "nix",
        "build",
        str(Path(repo) / "stack"),
        "--out-link",
        str(Path(repo) / "stack/result"),
    ]
    result = stream_run_logger(
        cmd=shlex.join(environment),
        cwd=repo,
        logger=logger,
        stdout_prefix="model environment",
        stderr_prefix="model environment",
        task_scope=task_scope,
    )
    if result.returncode:
        raise RuntimeError(f"Model environment build failed ({result.returncode})")
    from steps.stack.scripts.build import build_design

    # Models compile with the design's compiler plugins and link its device runtime.
    design_cmake = build_design(
        repo, design=design, selection=selection, logger=logger, task_scope=task_scope
    )
    for model in models:
        build = source / "build" / design / selection / model
        build_command = ["cmake", "--build", str(build), "--parallel", "1"]
        build_command.extend(["--target", f"{model}-model-build"])
        # A recipe reads the prepared model while it configures, so the model
        # is prepared first, by its own project: its weights and checkpoints
        # are rebuilt only when their inputs change.
        prepare = source / "build" / "prepare" / model
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
                f"-DARCH={arch}",
                f"-DDESIGN_CMAKE={design_cmake}",
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
