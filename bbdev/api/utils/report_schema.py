import math
import re
from datetime import datetime, timezone
from pathlib import Path

STATUSES = {"pass", "fail", "timeout", "infra_error"}
CATEGORIES = {
    "Conv",
    "MatMul",
    "Transfer",
    "Quantization",
    "Activation",
    "Pool",
    "Other",
}


def report_directory(run):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run["design"]):
        raise ValueError("design must be a safe directory name")
    timestamp = datetime.fromisoformat(run["timestamp"].replace("Z", "+00:00"))
    if timestamp.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    folder = run["id"]
    return Path(run["design"]) / folder


def nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def number(value, label, *, positive=False, integer=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite number")
    if (
        value < 0
        or (positive and value == 0)
        or (integer and (value != int(value) or value > 2**53 - 1))
    ):
        raise ValueError(f"invalid {label}: {value}")
    return value


def validate_run(run):
    if run["schema_version"] != 3:
        raise ValueError("invalid schema_version")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run["id"]):
        raise ValueError("run id must be a safe file name")
    stamp = datetime.fromisoformat(run["timestamp"].replace("Z", "+00:00"))
    if stamp.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", run["repository"]):
        raise ValueError("repository must be owner/name")
    if run["commit"] is not None and not re.fullmatch(r"[0-9a-f]{40}", run["commit"]):
        raise ValueError("commit must be a full 40-character SHA")
    for key in ("branch", "design", "backend", "config_hash", "producer"):
        nonempty(run[key], key)
    if run["clock_hz"] is not None:
        number(run["clock_hz"], "clock_hz", positive=True)
    if run["area"] is not None:
        area = run["area"]
        if area["status"] != "pass":
            raise ValueError("area measurement requires a successful DC result")
        number(area["cell_um2"], "cell_um2", positive=True)
        if not isinstance(area["constraints_met"], bool):
            raise ValueError("area constraints_met must be a boolean")
        for key in ("scope", "conditions", "report_url", "constraints_report_url"):
            nonempty(area[key], f"area {key}")
    if len(set(run["verification_targets"])) != len(run["verification_targets"]):
        raise ValueError("duplicate expected verification target")
    for target in run["verification_targets"]:
        nonempty(target, "expected verification target")
    targets = set()
    for result in run["verification"]:
        target = nonempty(result["target"], "verification target")
        if target in targets:
            raise ValueError(f"duplicate verification target: {target}")
        targets.add(target)
        if result["status"] not in STATUSES | {"not_run"}:
            raise ValueError("invalid verification status")
        if result["coverage"] is not None:
            if not result["coverage"]:
                raise ValueError("empty coverage measurements")
            for metric, value in result["coverage"].items():
                nonempty(metric, "coverage metric")
                if number(value, "coverage percent") > 100:
                    raise ValueError("coverage must be in [0,100]")
            nonempty(result["report_url"], "coverage report_url")
    if targets - set(run["verification_targets"]):
        raise ValueError("verification results contain undeclared targets")
    names = set()
    for model in run["workloads"]:
        if model["kind"] not in {"model", "core", "ball"}:
            raise ValueError("invalid workload kind")
        if model["latency_ms"] is not None:
            number(model["latency_ms"], "latency_ms", positive=True)
        if model["throughput"] is not None:
            number(model["throughput"]["value"], "throughput", positive=True)
            nonempty(model["throughput"]["unit"], "throughput unit")
        name = nonempty(model["name"], "model name")
        if name in names:
            raise ValueError(f"duplicate model: {name}")
        names.add(name)
        for key in ("workload_hash", "dataset_hash"):
            nonempty(model[key], key)
        if model["status"] not in STATUSES:
            raise ValueError(f"unknown status: {model['status']}")
        if model["accuracy"] is not None:
            accuracy = number(model["accuracy"], "accuracy")
            if accuracy > 1:
                raise ValueError("accuracy must be in [0,1]")
            nonempty(model["accuracy_metric"], "accuracy_metric")
        counter = model["counter"]
        if counter not in ("riscv-cycle", "rtl-cycle", "x86-tsc"):
            raise ValueError(f"unknown model counter: {counter}")
        count_field = "cycles" if counter != "x86-tsc" else "ticks"
        other = "ticks" if counter != "x86-tsc" else "cycles"
        if other in model:
            raise ValueError("model must not mix cycle and tick fields")
        if model["status"] == "pass":
            number(model[count_field], count_field, positive=True, integer=True)
            if model["trace_url"] is not None and not model["operators"]:
                raise ValueError("trace attachment requires operator measurements")
            if model["error"] is not None:
                raise ValueError("passing model cannot contain an error")
        else:
            nonempty(model["error"], "model error")
            if (
                model[count_field] is not None
                or model["operators"]
                or model["latency_ms"] is not None
                or model["throughput"] is not None
            ):
                raise ValueError(
                    "failed models cannot contain performance measurements"
                )
        ids = set()
        for op in model["operators"]:
            nonempty(op["id"], "operator id")
            nonempty(op["name"], "operator name")
            if op["id"] in ids:
                raise ValueError("duplicate operator id")
            ids.add(op["id"])
            if op["category"] not in CATEGORIES:
                raise ValueError("unknown operator category")
            if op["counter"] != counter or other in op:
                raise ValueError("operator counter differs from its model")
            unit = "cycle" if counter != "x86-tsc" else "tick"
            opposite = "tick" if unit == "cycle" else "cycle"
            if f"start_{opposite}" in op or f"end_{opposite}" in op:
                raise ValueError("operator mixes cycle and tick intervals")
            for key in (count_field, f"start_{unit}", f"end_{unit}", "level"):
                number(op[key], key, integer=True)
            number(op["calls"], "calls", positive=True, integer=True)
            if (
                op[f"start_{unit}"] + op[count_field] > op[f"end_{unit}"]
                or op[f"end_{unit}"] > model[count_field]
            ):
                raise ValueError("operator interval exceeds model trace span")
    if run["submission_errors"] != submission_errors(run):
        raise ValueError("submission status does not match the report measurements")
    return run


def submission_errors(run):
    errors = []
    if not run["workloads"] and not run["verification"] and run["area"] is None:
        errors.append("没有测量或验证结果")
    if run["commit"] is None and not run.get("provenance", {}).get("executable_sha256"):
        errors.append("缺少源码版本或执行产物哈希")
    return errors


def validate_submission(run):
    validate_run(run)
    if run["verification"] or run["verification_targets"]:
        raise ValueError(
            "verification results belong in test reports, not the leaderboard"
        )
    if run["backend"] == "dc":
        if run["area"] is None or run["workloads"]:
            raise ValueError(
                "DC submissions require an area measurement without model results"
            )
        if run["commit"] is None:
            raise ValueError("DC area report requires a source revision")
    else:
        if run["area"] is not None:
            raise ValueError(
                "area measurements must be submitted as a separate DC report"
            )
        if run["backend"] != "p2e" or not run["workloads"]:
            raise ValueError("leaderboard requires FPGA end-to-end model results")
        provenance = run["provenance"]
        for field in ("bitstream_sha256", "executable_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", provenance.get(field, "")):
                raise ValueError(f"FPGA model report requires {field}")
        for result in run["workloads"]:
            if (
                result["kind"] != "model"
                or result["benchmark"]["scope"] != "model_execution"
            ):
                raise ValueError(
                    "core, Ball and operator tests are not leaderboard model runs"
                )
            if result["counter"] != "riscv-cycle":
                raise ValueError(
                    "FPGA model performance requires the device cycle counter"
                )
            number(
                result["benchmark"]["samples"],
                "benchmark samples",
                positive=True,
                integer=True,
            )
            nonempty(result["report_url"], "raw end-to-end benchmark log")
    missing = submission_errors(run)
    if missing:
        raise ValueError(f"submission blocked for {run['id']}: {'; '.join(missing)}")
    return run
