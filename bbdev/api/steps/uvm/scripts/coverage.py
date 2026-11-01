import re
import shutil
from pathlib import Path

from .build import VerificationFailure, execute


def summary(directory):
    lines = (Path(directory) / "dashboard.txt").read_text().splitlines()
    start = lines.index("Total Coverage Summary ")
    for index in range(start + 1, len(lines)):
        if lines[index].strip().startswith("SCORE"):
            return {
                name.lower(): None if value == "--" else float(value)
                for name, value in zip(
                    lines[index].split(), lines[index + 1].split(), strict=True
                )
            }
    raise ValueError(f"Missing coverage summary: {directory}")


def validate(exclusions, exported):
    """Every reviewed exclusion must name an instance and checksum the tool exports now."""
    current = set()
    for path in Path(exported).glob("fullexclude.*"):
        for block in path.read_text().split('// CHECKSUM: "')[1:]:
            checksum = block.split('"', 1)[0]
            instance = re.search(r"^// INSTANCE: (\S+)", block, re.MULTILINE)
            if instance:
                current.add((instance[1], checksum))
    checksum = None
    for line in Path(exclusions).read_text().splitlines():
        if line.startswith("CHECKSUM:"):
            checksum = line.split('"')[1]
        elif line.startswith("INSTANCE:"):
            if (line.split()[1], checksum) not in current:
                raise ValueError(
                    f"coverage exclusions require review: {exclusions}: {line}"
                )
        elif line.startswith("MODULE:"):
            raise ValueError(f"exclusions must use instance scopes: {exclusions}")


def collect(repo, test, database, output, ctx, exclusions):
    """`dut` applies the test's reviewed exclusions when it has them; `dut_raw` never does."""
    output = Path(output)
    hierarchy = output.parent / "dut.cfg"
    hierarchy.write_text(
        "".join(f"+tree {test['top']}.{scope}\n" for scope in test["rtl_scopes"])
    )
    scope = ["-hier", hierarchy, "-metric", "+".join(test["metrics"])]
    report = ["urg", "-dir", database, "-format", "text"]
    commands = [
        ([*report, "-dump", "full_exclusions", "-report", output / "all"], "all"),
        ([*report, *scope, "-report", output / "dut_raw"], "dut_raw"),
    ]
    if exclusions.is_file():
        commands.append(
            (
                [*report, *scope, "-elfile", exclusions, "-excl_strict"]
                + ["-report", output / "dut"],
                "dut",
            )
        )
    for command, name in commands:
        if name == "dut":
            validate(exclusions, output.parent)
        logfile = output.parent / f"coverage-{name}.log"
        result = execute(repo, command, output.parent, logfile, 120, ctx)
        if result.returncode:
            raise VerificationFailure("coverage", test["name"], logfile)
        if name == "dut" and re.search(r"(?:Warning|Error)-\[", logfile.read_text()):
            raise VerificationFailure("coverage_exclusions", test["name"], logfile)
    if not exclusions.is_file():
        shutil.copytree(output / "dut_raw", output / "dut")
    values = summary(output / "dut")
    for metric in test["metrics"]:
        name = {"tgl": "toggle", "cond": "cond", "line": "line"}[metric]
        if values[name] is None:
            raise VerificationFailure(
                "coverage_metric", test["name"], output / "dut/dashboard.txt"
            )
    groups = {}
    if test["functional_groups"]:
        text = (output / "all/grpinfo.txt").read_text()
        for name in test["functional_groups"]:
            match = re.search(
                r"Group : " + re.escape(name) + r"\s.*?\nSCORE[^\n]*\n\s*([\d.]+)",
                text,
                re.DOTALL,
            )
            if match is None:
                raise VerificationFailure(
                    "functional_coverage", test["name"], output / "all/groups.txt"
                )
            groups[name] = float(match[1])
    return {
        "dut": values,
        "dut_raw": summary(output / "dut_raw"),
        "functional": groups,
        "report": str(output),
    }
