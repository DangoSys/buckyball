from pathlib import Path

from utils.reports import new_report, attachment, finish_report
from utils.path import load_design_config
from .uvm_common import dashboard_summary, load_ip


def verification_targets(repo, design, ball=None, ip=None):
    if ip is not None:
        return [
            f"ip/{ip}/{item['name']}"
            for profile in load_ip(str(repo), ip, design)
            for item in profile["targets"]
        ]
    if ball is not None:
        return [f"ball/{ball}"]
    config = load_design_config(repo, design).verification
    targets = [f"ball/{ball}" for ball in config.balls]
    for ip in config.ips:
        targets.extend(
            f"ip/{ip}/{target['name']}"
            for profile in load_ip(str(repo), ip, design=design)
            for target in profile["targets"]
        )
    return targets


def coverage_report(context, trace_id, info, output, expected_targets):
    report = new_report(context, "uvm.run", trace_id, "uvm")
    for result in info["results"]:
        if "balls" in result:
            rows = [
                (
                    f"ball/{name}",
                    Path(result["log"])
                    / (f"{name}/coverage" if len(result["balls"]) > 1 else "coverage"),
                )
                for name in result["balls"]
            ]
        else:
            rows = [
                (
                    f"ip/{result['ip']}/{target}",
                    Path(result["log"]) / target / "coverage",
                )
                for target in result["targets"]
            ]
        for target, directory in rows:
            report["verification_targets"].append(target)
            if (
                result.get("expected_assertions")
                and target.removeprefix(f"ip/{result['ip']}/")
                in result["expected_assertions"]
            ):
                report["verification"].append(
                    {
                        "target": target,
                        "status": "pass",
                        "coverage": None,
                        "report_url": None,
                    }
                )
                continue
            values = dashboard_summary(str(directory))
            url = attachment(
                report,
                output,
                directory / "dashboard.txt",
                f"coverage-{target.replace('/', '-')}.txt",
            )
            report["verification"].append(
                {
                    "target": target,
                    "status": "pass",
                    "coverage": {
                        name: float(value)
                        for name, value in values.items()
                        if value != "--"
                    },
                    "report_url": url,
                }
            )
    for failure in info["failures"]:
        target = failure["target"]
        report["verification_targets"].append(target)
        report["verification"].append(
            {
                "target": target,
                "status": "fail",
                "coverage": None,
                "report_url": None,
                "error": failure["error"],
            }
        )
    report["verification_targets"] = expected_targets
    return finish_report(report, output)
