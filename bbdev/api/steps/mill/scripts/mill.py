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
        # A simulation entry (devices/<device>/sims/<entry>/arch, harts/<hart>/
        # sims/hart/arch) builds in the design's output/sims/<entry>/<name>/mill:
        # each hart of a design has its own entry.
        parts = directory.relative_to(designs).parts
        owner = designs / parts[0]
        if "sims" in parts:
            sims = parts.index("sims")
            work = owner.joinpath(
                "output/sims", *parts[sims + 1 : -1], parts[sims - 1], "mill"
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
            (
                # A thread alone runs on its virtual environment; its entry
                # class elaborates the thread alone, for synthesis.
                "framework.thread.sims.Verilator"
                if selection.startswith("thread/") and product == "verilator"
                # A Ball or a thread alone: its entry class is its elaborator.
                else (
                    registered.entry_class
                    if selection.startswith(("ball/", "thread/"))
                    else (
                        "framework.device.sims.p2e.Elaborate"
                        if product == "p2e"
                        else "framework.device.sims.verilator.Elaborate"
                    )
                )
            ),
            "--config",
            str(config),
            "--target-class",
            registered.entry_class,
            "--target-dir",
            out,
            "--disable-annotation-unknown",
            "--strip-debug-info",
            "-O=debug",
            "-lowering-options=disallowLocalVariables",
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
                "--repl-seq-mem",
                f"--repl-seq-mem-file={os.path.join(out, 'mems.conf')}",
            ]
        )
    return command
