from pathlib import Path

from utils.reports import new_report, finish_report


def performance_report(repo, context, trace_id, backend, log, stdout, returncode):
    report = new_report(context, f"{backend}.execution", trace_id, backend)
    report["execution"] = {
        "status": "pass" if returncode == 0 else "fail",
        "returncode": returncode,
        "performance": "not_collected",
    }
    return finish_report(report, Path(log) / "report")
