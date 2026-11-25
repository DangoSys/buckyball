import hashlib
import re
from pathlib import Path

from utils.reports import new_report, attachment, finish_report


def area_report(context, trace_id, report_dir, contract):
    report_dir = Path(report_dir)
    output = report_dir.parent / "report"
    report = new_report(context, "dc.area", trace_id, "dc")
    values = re.findall(
        r"Total cell area:\s*([0-9.eE+-]+)", (report_dir / "area.rpt").read_text()
    )
    if len(values) != 1:
        raise ValueError("DC report must contain one total cell area")
    # Every Ball: a direct child of a BBus whose module is a Ball class
    # (<Name>Ball, uniquified <Name>Ball_<n>), with its total area (the
    # absolute total, then percent, combinational, ...).
    balls = [
        {"instance": m.group(1), "module": m.group(3), "cell_um2": float(m.group(2))}
        for m in re.finditer(
            r"(?m)^((?:\S+/)?bbus/[^/\s]+)\s+([0-9.eE+-]+)\s+\S+\s+\S+\s+\S+\s+\S+\s+(\w+Ball(?:_\d+)?)\s*$",
            (report_dir / "area.rpt").read_text(),
        )
    ]
    constraints = (report_dir / "constraint.rpt").read_text()
    if "Report : constraint" not in constraints:
        raise ValueError("invalid DC constraint report")
    report["area"] = {
        "status": "pass",
        "cell_um2": float(values[0]),
        "scope": contract.top_module,
        "conditions": (
            f"DC · {Path(contract.target_library).name} · library SHA256 "
            f"{hashlib.sha256(Path(contract.target_library).read_bytes()).hexdigest()} · SDC "
            f"{hashlib.sha256(contract.constraints_sdc.read_bytes()).hexdigest()}"
        ),
        "constraints_met": "(VIOLATED)" not in constraints,
        "balls": balls,
        "report_url": attachment(report, output, report_dir / "area.rpt", "area.rpt"),
        "constraints_report_url": attachment(
            report, output, report_dir / "constraint.rpt", "area-constraints.rpt"
        ),
    }
    return finish_report(report, output)
