import fcntl
import json
import shlex
import shutil
from pathlib import Path

from steps.mill.scripts.mill import mill_cmd
from utils.stream_run import stream_run_logger


class VerificationFailure(RuntimeError):
    def __init__(self, stage, test, logfile):
        self.stage, self.test, self.logfile = stage, test, str(logfile)
        super().__init__(f"{stage} failed: {test}; see {logfile}")


def execute(repo, arguments, cwd, logfile, timeout, ctx):
    logfile = Path(logfile)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "nix",
        "develop",
        str(Path(repo) / "verify"),
        "--command",
        "timeout",
        "--kill-after=5s",
        f"{timeout}s",
        *map(str, arguments),
    ]
    result = stream_run_logger(
        command,
        ctx.logger,
        cwd=str(cwd),
        shell=False,
        timeout=timeout + 30,
        stdout_prefix="verification",
        stderr_prefix="verification",
    )
    logfile.write_text(result.stdout + "\n" + result.stderr)
    return result


def build(repo, design, profile, logdir, ctx):
    repo = Path(repo)
    root = repo / profile["root"]
    scope = design if design is not None else "default"
    output = root / "build/verification" / scope / profile["profile"]
    output.mkdir(parents=True, exist_ok=True)
    lock = (root / "build/.verification.lock").open("w")
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        model = profile["model"]
        if model["manifest"]:
            result = execute(
                repo,
                [
                    "cargo",
                    "build",
                    "--manifest-path",
                    root / model["manifest"],
                    "--target-dir",
                    root / "build/verification/reference",
                ],
                root,
                Path(logdir) / "reference.log",
                600,
                ctx,
            )
            if result.returncode:
                raise VerificationFailure(
                    "reference", "", Path(logdir) / "reference.log"
                )
        parameters = output / "parameters.json"
        parameters.write_text(json.dumps(profile["parameters"]))
        rtl = output / "rtl"
        registration = profile["build"]
        if registration["kind"] != "mill":
            raise ValueError(f"Unknown emitter kind: {registration['kind']}")
        replacements = {
            "{repo}": str(repo),
            "{design}": design,
            "{config}": str(parameters),
            "{rtl}": str(rtl),
            "{targets}": ",".join(test["name"] for test in profile["tests"]),
        }
        arguments = []
        for argument in registration["arguments"]:
            for token, value in replacements.items():
                if token in argument:
                    if value is None:
                        raise ValueError(
                            f"{profile['ip']} requires a design for {token}"
                        )
                    argument = argument.replace(token, value)
            arguments.append(argument)
        if profile["parameters"] and not any(
            "{config}" in value for value in registration["arguments"]
        ):
            raise ValueError(f"Emitter does not consume parameters: {profile['ip']}")
        command = mill_cmd(
            repo,
            registration["directory"],
            registration["module"] + ".run",
            arguments,
        )
        result = execute(
            repo,
            ["zsh", "-c", command],
            repo / registration["directory"],
            Path(logdir) / "emit.log",
            900,
            ctx,
        )
        if result.returncode:
            raise VerificationFailure("emit", "", Path(logdir) / "emit.log")
        if not any("{rtl}" in value for value in registration["arguments"]):
            shutil.copytree(
                root / "build",
                rtl,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    "verification", "uvm", ".verification.lock", ".build.lock"
                ),
            )
        resources = root / profile["resources"]
        compiled = {}
        binaries = {}
        for test in profile["tests"]:
            directory = output / test["name"]
            directory.mkdir(parents=True, exist_ok=True)
            source = resources / test["filelist"]
            filelist = directory / "filelist.f"
            filelist.write_text(
                source.read_text()
                .replace("@VERIFY@", str(repo / "verify"))
                .replace("@RESOURCES@", str(source.parent))
                .replace("@RTL@", str(rtl))
            )
            signature = (
                filelist.read_text(),
                test["top"],
                tuple(test["metrics"]),
                tuple(test["rtl_scopes"]),
            )
            if signature in compiled:
                binaries[test["name"]] = compiled[signature]
                continue
            hierarchy = directory / "coverage.cfg"
            hierarchy.write_text(
                "".join(
                    f"+tree {test['top']}.{scope}\n" for scope in test["rtl_scopes"]
                )
            )
            metrics = "+".join([*test["metrics"], "assert"])
            command = [
                "uvm-compile",
                "-top",
                test["top"],
                "-debug_access+all",
                "-f",
                filelist,
                "-o",
                directory / "simv",
                "-Mdir=" + str(directory / "csrc"),
                *[
                    f"+define+{name}={value}"
                    for name, value in profile["defines"].items()
                ],
            ]
            if test["metrics"]:
                command.extend(["-cm", metrics, "-cm_hier", hierarchy])
            result = execute(
                repo,
                command,
                directory,
                Path(logdir) / test["name"] / "compile.log",
                600,
                ctx,
            )
            if result.returncode:
                raise VerificationFailure(
                    "compile", test["name"], Path(logdir) / test["name"] / "compile.log"
                )
            compiled[signature] = directory / "simv"
            binaries[test["name"]] = directory / "simv"
    return {"output": output, "binaries": binaries}
