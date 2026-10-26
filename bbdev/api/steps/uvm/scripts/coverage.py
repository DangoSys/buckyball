import re
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


def collect(repo, test, database, output, ctx):
    output = Path(output)
    hierarchy = output.parent / "dut.cfg"
    hierarchy.write_text(
        "".join(f"+tree {test['top']}.{scope}\n" for scope in test["rtl_scopes"])
    )
    commands = [
        (
            ["urg", "-dir", database, "-format", "text", "-report", output / "all"],
            "all",
        ),
        (
            [
                "urg",
                "-dir",
                database,
                "-format",
                "text",
                "-hier",
                hierarchy,
                "-metric",
                "+".join(test["metrics"]),
                "-report",
                output / "dut",
            ],
            "dut",
        ),
    ]
    for command, name in commands:
        logfile = output.parent / f"coverage-{name}.log"
        result = execute(repo, command, output.parent, logfile, 120, ctx)
        if result.returncode:
            raise VerificationFailure("coverage", test["name"], logfile)
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
    return {"dut": values, "functional": groups, "report": str(output)}
