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
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run["chip"]):
        raise ValueError("chip must be a safe directory name")
    timestamp = datetime.fromisoformat(run["timestamp"].replace("Z", "+00:00"))
    if timestamp.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    folder = timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d-%H-%M")
    return Path(run["chip"]) / folder


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
    if run["schema_version"] != 2:
        raise ValueError("invalid schema_version")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run["id"]):
        raise ValueError("run id must be a safe file name")
    stamp = datetime.fromisoformat(run["timestamp"].replace("Z", "+00:00"))
    if stamp.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", run["repository"]):
        raise ValueError("repository must be owner/name")
    if not re.fullmatch(r"[0-9a-f]{40}", run["commit"]):
        raise ValueError("commit must be a full 40-character SHA")
    for key in ("branch", "chip", "backend", "config_hash", "producer"):
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
    for model in run["models"]:
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
        if model["status"] == "pass":
            number(model["cycles"], "cycles", positive=True, integer=True)
            if model["trace_url"] is not None and not model["operators"]:
                raise ValueError("trace attachment requires operator measurements")
            if model["error"] is not None:
                raise ValueError("passing model cannot contain an error")
        else:
            nonempty(model["error"], "model error")
            if model["cycles"] is not None or model["operators"]:
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
            for key in ("cycles", "start_cycle", "level"):
                number(op[key], key, integer=True)
            number(op["calls"], "calls", positive=True, integer=True)
            if op["start_cycle"] + op["cycles"] > model["cycles"]:
                raise ValueError("operator interval exceeds model trace span")
    if run["submission_errors"] != submission_errors(run):
        raise ValueError("submission status does not match the report measurements")
    return run


def submission_errors(run):
    missing = []
    if run["area"] is None:
        missing.append("Area：缺少面积报告")
    elif not run["area"]["constraints_met"]:
        missing.append("Area：综合约束未满足")
    if not run["models"]:
        missing.append("Performance：没有模型结果")
    for model in run["models"]:
        if model["status"] != "pass":
            missing.append(f"Performance：{model['name']} 未通过")
        for key, label in (
            ("accuracy", "精度"),
            ("latency_ms", "延迟"),
            ("throughput", "吞吐"),
        ):
            if model[key] is None:
                missing.append(f"Performance：{model['name']} 缺少{label}")
    expected = set(run["verification_targets"])
    actual = {result["target"] for result in run["verification"]}
    if not expected or expected != actual:
        missing.append("Coverage：验证项不完整")
    for result in run["verification"]:
        if result["status"] != "pass" or result["coverage"] is None:
            missing.append(f"Coverage：{result['target']} 未通过或缺少覆盖率报告")
    if run.get("provenance", {}).get("source_dirty", False):
        missing.append("源码包含未提交修改")
    return missing


def validate_submission(run):
    validate_run(run)
    missing = submission_errors(run)
    if missing:
        raise ValueError(f"submission blocked for {run['id']}: {'; '.join(missing)}")
    return run
