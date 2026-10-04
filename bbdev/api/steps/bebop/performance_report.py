import hashlib
import importlib.util
import json
import re
import tomllib
from pathlib import Path

from utils.reports import new_report, attachment, finish_report
from utils.report_schema import number, nonempty, CATEGORIES
from bisect import bisect_right


def parse_trace(data, categories):
    events = data["traceEvents"]
    groups = {}
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "buddy.trace":
            continue
        args = event["args"]
        path = args["id_path"]
        if not isinstance(path, list) or not path:
            raise ValueError("trace id_path must be a non-empty list")
        for part in path:
            number(part, "trace id", integer=True)
        identity = "-".join(str(int(part)) for part in path)
        start = number(args["start_cycle"], "start_cycle", integer=True)
        end = number(args["end_cycle"], "end_cycle", integer=True)
        elapsed = number(args["elapsed_cycle"], "elapsed_cycle", integer=True)
        if end < start or end - start != elapsed or event["dur"] != elapsed:
            raise ValueError(f"inconsistent cycle interval for {identity}")
        name = nonempty(event["name"], "operator name")
        category = categories.get(name, "Other")
        if category not in CATEGORIES:
            raise ValueError(f"unknown operator category: {category}")
        groups.setdefault(identity, []).append(
            {
                "id": identity,
                "name": name,
                "category": category,
                "cycles": int(elapsed),
                "start_cycle": int(start),
                "level": len(path) - 1,
            }
        )
    if not groups:
        raise ValueError("no buddy.trace cycle intervals found")
    starts = {}
    for identity, records in groups.items():
        records.sort(key=lambda op: op["start_cycle"])
        starts[identity] = [op["start_cycle"] for op in records]
        for previous, current in zip(records, records[1:]):
            if previous["name"] != current["name"]:
                raise ValueError(f"trace id has inconsistent names: {identity}")
            if previous["start_cycle"] + previous["cycles"] > current["start_cycle"]:
                raise ValueError(f"overlapping calls of trace {identity}")
    for identity, records in groups.items():
        parts = identity.split("-")
        if len(parts) > 1:
            parent_id = "-".join(parts[:-1])
            if parent_id not in groups:
                raise ValueError(f"missing parent for trace {identity}")
            for op in records:
                index = bisect_right(starts[parent_id], op["start_cycle"]) - 1
                if index < 0:
                    raise ValueError(f"nested trace outside parent: {identity}")
                parent = groups[parent_id][index]
                if (
                    op["start_cycle"] + op["cycles"]
                    > parent["start_cycle"] + parent["cycles"]
                ):
                    raise ValueError(f"nested trace outside parent: {identity}")
    start = min(records[0]["start_cycle"] for records in groups.values())
    end = max(
        records[-1]["start_cycle"] + records[-1]["cycles"]
        for records in groups.values()
    )
    if start == end:
        raise ValueError("trace span must be positive")
    operators = []
    for records in groups.values():
        op = dict(records[0])
        op["cycles"] = sum(record["cycles"] for record in records)
        op["calls"] = len(records)
        op["start_cycle"] -= start
        operators.append(op)
    return int(end - start), operators


def performance_report(repo, context, trace_id, backend, log, stdout, returncode):
    repo, log = Path(repo), Path(log)
    records = [
        json.loads(line.split("@BBPERF ", 1)[1])
        for line in stdout.splitlines()
        if "@BBPERF " in line
    ]
    if not records:
        report = new_report(context, f"{backend}.execution", trace_id, backend)
        report["execution"] = {
            "status": "pass" if returncode == 0 else "fail",
            "returncode": returncode,
            "performance": "not_collected",
        }
        return finish_report(report, log / "report")
    models = {record["model"] for record in records}
    if len(models) != 1:
        raise ValueError("one model is required per performance run")
    if len({record["unit"] for record in records}) != 1:
        raise ValueError("benchmark throughput units differ")
    model = models.pop()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", model):
        raise ValueError("invalid benchmark model name")
    recipe = repo / f"examples/models/{context['chip']}/{model}"
    settings = tomllib.loads((recipe / "configs/report.toml").read_text())
    report = new_report(context, f"{backend}.performance", trace_id, backend)
    output = log / "report"
    report["clock_hz"] = settings.get(backend, {}).get("clock_hz")
    expected = settings.get("expected_classes")
    correct = samples = elapsed = cycles = 0
    for index, record in enumerate(records):
        number(record["samples"], "samples", positive=True, integer=True)
        number(record["elapsed_ns"], "elapsed_ns", positive=True, integer=True)
        number(record["cycles"], "cycles", positive=True, integer=True)
        samples += record["samples"]
        elapsed += record["elapsed_ns"]
        cycles += record["cycles"]
        if "correct" in record:
            number(record["correct"], "correct samples", integer=True)
            if record["correct"] > record["samples"]:
                raise ValueError("correct samples exceed sample count")
            correct += record["correct"]
        elif expected is not None:
            if len(expected) != len(records):
                raise ValueError("reference labels do not match the benchmark inputs")
            if sum(record["predictions"].values()) != record["samples"]:
                raise ValueError("prediction counts do not match sample count")
            correct += record["predictions"].get(str(expected[index]), 0)
    accuracy = (
        correct / samples
        if expected is not None or all("correct" in record for record in records)
        else None
    )
    build = repo / f"stack/models/build/{context['chip']}/{model}"
    workload = hashlib.sha256()
    for file in sorted((build / "artifact").rglob("*")):
        if file.is_file():
            workload.update(file.relative_to(build / "artifact").as_posix().encode())
            workload.update(file.read_bytes())
    run_config = recipe / "configs/running-param.toml"
    inputs = tomllib.loads(run_config.read_text())["inputs"]
    dataset = hashlib.sha256()
    for value in inputs:
        if isinstance(value, str) and (run_config.parent / value).is_file():
            dataset.update((run_config.parent / value).read_bytes())
        else:
            dataset.update(json.dumps(value, sort_keys=True).encode())
    operators = {}
    all_events = []
    trace_span = 0
    trace_roots = sorted(
        {
            file.parent.parent
            for file in log.rglob("trace-*.txt")
            if file.parent.name == "cycle"
        }
    )
    if trace_roots:
        spec = importlib.util.spec_from_file_location(
            "bbdev_perfetto", repo / "stack/serving/trace/perfetto.py"
        )
        perfetto = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(perfetto)
        metadata = [build / f"models/{model}/subgraph0.trace.mlir"]
        for index, root in enumerate(trace_roots):
            data = perfetto.build_perfetto(root, recipe / "trace/trace.toml", metadata)
            span, ops = parse_trace(data, settings.get("operator_categories", {}))
            trace_span += span
            for op in ops:
                if op["id"] in operators:
                    operators[op["id"]]["cycles"] += op["cycles"]
                    operators[op["id"]]["calls"] += op["calls"]
                else:
                    operators[op["id"]] = op
            for event in data["traceEvents"]:
                event["pid"] = index
            all_events.extend(data["traceEvents"])
    result = {
        "name": model,
        "status": "pass" if returncode == 0 else "fail",
        "workload_hash": workload.hexdigest(),
        "dataset_hash": dataset.hexdigest(),
        "cycles": trace_span if trace_roots else cycles,
        "accuracy": accuracy,
        "accuracy_metric": records[0].get(
            "accuracy_metric", settings["accuracy_metric"]
        ),
        "latency_ms": elapsed / samples / 1e6,
        "throughput": {"value": samples * 1e9 / elapsed, "unit": records[0]["unit"]},
        "error": None,
        "trace_url": None,
        "operators": list(operators.values()),
        "benchmark": {
            "samples": samples,
            "elapsed_ns": elapsed,
            "scope": "model_execution",
        },
    }
    if returncode != 0:
        result.update(
            cycles=None,
            accuracy=None,
            latency_ms=None,
            throughput=None,
            operators=[],
            error=f"model exited with {returncode}",
        )
    elif all_events:
        path = log / "perfetto.json"
        path.write_text(json.dumps({"traceEvents": all_events}) + "\n")
        result["trace_url"] = attachment(
            report, output, path, f"model-{model}-perfetto.json"
        )
    report["models"].append(result)
    return finish_report(report, output)
