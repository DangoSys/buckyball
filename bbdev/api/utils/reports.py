import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from utils.path import _design_name

from utils.report_schema import (
    report_directory,
    submission_errors,
    validate_run,
)


def source_context(repo, design):
    repo = Path(repo)
    design = _design_name(design)
    config = (
        repo / "examples/designs" / design / "output/generated/design.pb"
    ).read_bytes()

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    return {
        "repository": "DangoSys/buckyball",
        "branch": git("branch", "--show-current") or git("rev-parse", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "design": design,
        "config_hash": hashlib.sha256(config).hexdigest(),
        "provenance": {
            "source_dirty": bool(
                git(
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                    "--ignore-submodules=none",
                )
            ),
            "workflow_url": None,
            "comparable": True,
        },
    }


def new_report(context, producer, trace_id, backend):
    now = datetime.now(timezone.utc)
    return {
        "schema_version": 3,
        "id": f"{context['design']}-{producer.replace('.', '-')}-{trace_id}",
        "timestamp": now.isoformat().replace("+00:00", "Z"),
        **context,
        "producer": producer,
        "backend": backend,
        "clock_hz": None,
        "area": None,
        "verification_targets": [],
        "verification": [],
        "workloads": [],
        "artifacts": {},
    }


def attachment(report, directory, source, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(source).resolve(strict=True)
    target = directory / name
    if target.exists():
        raise FileExistsError(target)
    shutil.copyfile(source, target)
    report["artifacts"][name] = hashlib.sha256(target.read_bytes()).hexdigest()
    return f"data/{report_directory(report).as_posix()}/{name}"


def finish_report(report, directory):
    report["submission_errors"] = submission_errors(report)
    validate_run(report)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "report.json"
    temporary = directory / ".report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)
    return str(path)


def verify_artifacts(report, directory):
    prefix = f"data/{report_directory(report).as_posix()}/"
    references = []
    if report["area"] is not None:
        references.extend(
            [report["area"]["report_url"], report["area"]["constraints_report_url"]]
        )
    references.extend(
        row["report_url"]
        for row in report["verification"]
        if row["report_url"] is not None
    )
    references.extend(
        model[field]
        for model in report["workloads"]
        for field in ("trace_url", "report_url")
        if model[field] is not None
    )
    for url in references:
        if not url.startswith(prefix):
            raise ValueError("attachment URL does not belong to this report")
        name = url[len(prefix) :]
        if name not in report["artifacts"]:
            raise ValueError(f"unregistered report attachment: {name}")
    for name, digest in report["artifacts"].items():
        if Path(name).name != name:
            raise ValueError("attachment name must be a file name")
        if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"attachment hash mismatch: {name}")
