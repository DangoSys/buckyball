from __future__ import annotations

import os
import shlex
from utils.path import get_buckyball_path, load_design_config, simulation_target
from pathlib import Path


def mill_cmd(
    repo: str, directory: str | Path, target: str, arguments=(), *, show: bool = False
) -> str:
    repo = Path(repo).resolve()
    directory = (repo / directory).resolve()
    designs = repo / "examples/designs"
    if directory.is_relative_to(designs):
        relative = directory.relative_to(designs)
        owner = designs / relative.parts[0]
        if directory.is_relative_to(owner / "sims"):
            work = (
                owner
                / "output/sims"
                / directory.relative_to(owner / "sims").parent
                / "mill"
            )
        else:
            work = owner / "output/mill"
    else:
        owner = (
            directory.parent
            if directory.name == "arch" and directory.is_relative_to(repo / "examples")
            else directory
        )
        work = owner / "output/mill"
    work.mkdir(parents=True, exist_ok=True)
    relative = Path(os.path.relpath(directory, work))
    imported = ".".join("^" if part == ".." else f"`{part}`" for part in relative.parts)
    source = f"import $file.{imported}.{{build => source}}\n\ndef project = source\n"
    launcher = work / "build.sc"
    if not launcher.is_file() or launcher.read_text() != source:
        launcher.write_text(source)
    command = shlex.join(
        [
            "flock",
            "/tmp/buckyball-mill.lock",
            "mill",
            "--no-server",
            *(["show"] if show else []),
            f"project.{target}",
            *arguments,
        ]
    )
    return f"cd {shlex.quote(str(work))} && {command}"


def design_cmd(
    repo: str,
    design: str,
    product: str,
    out: str,
    *,
    selection: str,
    diff: bool = False,
) -> tuple[str, str]:
    registered = simulation_target(repo, design, selection)
    build = registered.arch
    directory = Path(repo) / build.directory
    config = Path(repo) / build.config
    command = mill_cmd(
        repo,
        directory,
        build.module + ".runMain",
        [
            "sims.p2e.Elaborate" if product == "p2e" else "sims.verilator.Elaborate",
            "--config",
            str(config),
            "--target-class",
            registered.entry_class,
            "--target-dir",
            out,
            "--disable-annotation-unknown",
            "--strip-debug-info",
            "-O=debug",
            *(["--difftest"] if diff else []),
        ],
    )
    return command, str(directory)


def elaborate_cmd(
    design: str, selection: str, product: str, out: str, *, seq_mem: bool = False
) -> str:
    command, _ = design_cmd(
        get_buckyball_path(), design, product, out, selection=selection
    )
    if seq_mem:
        command += " " + shlex.join(
            [
                "-lowering-options=disallowLocalVariables",
                "--repl-seq-mem",
                f"--repl-seq-mem-file={os.path.join(out, 'mems.conf')}",
            ]
        )
    return command
