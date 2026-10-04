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
    constraints = (report_dir / "constraint.rpt").read_text()
    if "Report : constraint" not in constraints:
        raise ValueError("invalid DC constraint report")
    report["area"] = {
        "status": "pass",
        "cell_um2": float(values[0]),
        "scope": contract.top_module,
        "conditions": f"DC · {Path(contract.target_library).name} · SDC {hashlib.sha256(contract.constraints_sdc.read_bytes()).hexdigest()}",
        "constraints_met": "(VIOLATED)" not in constraints,
        "report_url": attachment(report, output, report_dir / "area.rpt", "area.rpt"),
        "constraints_report_url": attachment(
            report, output, report_dir / "constraint.rpt", "area-constraints.rpt"
        ),
    }
    return finish_report(report, output)
