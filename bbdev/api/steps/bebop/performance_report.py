import hashlib
import importlib.util
import json
import re
import tomllib
from pathlib import Path

from utils.reports import new_report, attachment, finish_report
from utils.report_schema import number, nonempty, CATEGORIES


def parse_trace(data, categories):
    groups = {}
    for event in data["traceEvents"]:
        if event.get("ph") != "X" or event.get("cat") != "buddy.trace":
            continue
        args = event["args"]
        counter = args["counter"]
        if counter not in ("riscv-cycle", "x86-tsc"):
            raise ValueError(f"invalid trace counter: {counter}")
        unit = "cycle" if counter == "riscv-cycle" else "tick"
        path = args["id_path"]
        if not isinstance(path, list) or not path:
            raise ValueError("trace id_path must be a non-empty list")
        for part in path:
            number(part, "trace id", integer=True)
        controller = int(number(args["controller"], "controller", integer=True))
        core = int(number(args["core"], "core", integer=True))
        platform = args["platform"]
        if platform not in ("linux", "baremetal"):
            raise ValueError(f"invalid trace platform: {platform}")
        number(event["pid"], "pid", integer=True, positive=platform == "linux")
        number(event["tid"], "tid", integer=True, positive=platform == "linux")
        if platform == "baremetal" and event["pid"] != 0:
            raise ValueError("baremetal trace requires pid=0")
        key = (controller, core, tuple(path))
        identity = f"controller-{controller}/core-{core}/" + "-".join(map(str, path))
        start, end = args[f"start_{unit}"], args[f"end_{unit}"]
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value >= 2**64
            for value in (start, end)
        ):
            raise ValueError("counter boundaries must be unsigned 64-bit integers")
        elapsed = number(args[f"elapsed_{unit}"], "elapsed_cycle", integer=True)
        if end < start or end - start != elapsed or event["dur"] != elapsed:
            raise ValueError(f"inconsistent cycle interval for {identity}")
        name = nonempty(event["name"], "operator name")
        category = categories.get(name, "Other")
        if category not in CATEGORIES:
            raise ValueError(f"unknown operator category: {category}")
        groups.setdefault(key, []).append(
            {
                "id": identity,
                "name": name,
                "category": category,
                "counter": counter,
                "duration": int(elapsed),
                "start": int(start),
                "end": int(end),
                "level": len(path) - 1,
            }
        )
    if not groups:
        raise ValueError("no buddy.trace cycle intervals found")
    if (
        len({record["counter"] for records in groups.values() for record in records})
        != 1
    ):
        raise ValueError("performance trace cannot mix counter sources")
    for key, records in groups.items():
        records.sort(key=lambda op: op["start"])
        for previous, current in zip(records, records[1:]):
            if previous["name"] != current["name"]:
                raise ValueError(f"trace scope has inconsistent names: {key}")
            if previous["start"] + previous["duration"] > current["start"]:
                raise ValueError(f"overlapping calls of trace scope {key}")
    for (controller, core, path), records in groups.items():
        if len(path) == 1:
            continue
        for op in records:
            parents = [
                parent
                for (owner, _, parent_path), calls in groups.items()
                if owner == controller and parent_path == path[:-1]
                for parent in calls
                if parent["start"] <= op["start"]
                and op["start"] + op["duration"] <= parent["start"] + parent["duration"]
            ]
            if len(parents) != 1:
                raise ValueError(
                    f"trace scope {controller}/{core}/{path} has {len(parents)} parents"
                )
    start = min(record["start"] for records in groups.values() for record in records)
    end = max(
        record["start"] + record["duration"]
        for records in groups.values()
        for record in records
    )
    if start == end:
        raise ValueError("trace span must be positive")
    operators = []
    for records in groups.values():
        first = records[0]
        unit = "cycle" if first["counter"] == "riscv-cycle" else "tick"
        plural = "cycles" if unit == "cycle" else "ticks"
        operators.append(
            {
                "id": first["id"],
                "name": first["name"],
                "category": first["category"],
                "counter": first["counter"],
                "level": first["level"],
                "calls": len(records),
                plural: sum(record["duration"] for record in records),
                f"start_{unit}": first["start"] - start,
                f"end_{unit}": records[-1]["end"] - start,
            }
        )
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
    counters = {record["counter"] for record in records}
    if len(counters) != 1 or not counters <= {"riscv-cycle", "x86-tsc"}:
        raise ValueError("benchmark records require one explicit counter source")
    benchmark_counter = counters.pop()
    benchmark_count_field = "cycles" if benchmark_counter == "riscv-cycle" else "ticks"
    correct = samples = elapsed = measured = 0
    for index, record in enumerate(records):
        number(record["samples"], "samples", positive=True, integer=True)
        number(record["elapsed_ns"], "elapsed_ns", positive=True, integer=True)
        number(
            record[benchmark_count_field],
            benchmark_count_field,
            positive=True,
            integer=True,
        )
        samples += record["samples"]
        elapsed += record["elapsed_ns"]
        other = "ticks" if benchmark_count_field == "cycles" else "cycles"
        if other in record:
            raise ValueError("benchmark record mixes cycle and tick fields")
        measured += record[benchmark_count_field]
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
    trace_counter = None
    trace_roots = sorted(
        {
            file.parents[3]
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
        for root in trace_roots:
            data = perfetto.build_perfetto(root, recipe / "trace/trace.toml", metadata)
            span, ops = parse_trace(data, settings.get("operator_categories", {}))
            counter = ops[0]["counter"]
            if trace_counter is not None and trace_counter != counter:
                raise ValueError("performance roots cannot mix counter sources")
            if counter != benchmark_counter:
                raise ValueError("benchmark and trace counter sources differ")
            trace_counter = counter
            count_field = "cycles" if counter == "riscv-cycle" else "ticks"
            offset = trace_span
            trace_span += span
            unit = "cycle" if counter == "riscv-cycle" else "tick"
            for op in ops:
                op[f"start_{unit}"] += offset
                op[f"end_{unit}"] += offset
                if op["id"] in operators:
                    operators[op["id"]][count_field] += op[count_field]
                    operators[op["id"]]["calls"] += op["calls"]
                    operators[op["id"]][f"end_{unit}"] = op[f"end_{unit}"]
                else:
                    operators[op["id"]] = op
            all_events.extend(data["traceEvents"])
    counter = trace_counter if trace_roots else benchmark_counter
    count_field = "cycles" if counter == "riscv-cycle" else "ticks"
    result = {
        "counter": counter,
        "name": model,
        "status": "pass" if returncode == 0 else "fail",
        "workload_hash": workload.hexdigest(),
        "dataset_hash": dataset.hexdigest(),
        count_field: trace_span if trace_roots else measured,
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
            **{count_field: None},
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
