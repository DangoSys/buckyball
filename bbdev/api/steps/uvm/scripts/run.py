import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from utils.path import log_dir
from .build import build, execute, VerificationFailure
from .config import profiles
from .coverage import collect


def judge(test, process, result_file):
    text = process.stdout + "\n" + process.stderr
    logfile = result_file.parent / "simulation.log"
    if process.returncode in (124, 137):
        raise VerificationFailure("wall_timeout", test["name"], logfile)
    if "[PH_TIMEOUT]" in text:
        raise VerificationFailure("simulation_timeout", test["name"], logfile)
    severities = dict(re.findall(r"UVM_(ERROR|FATAL)\s*:\s*(\d+)", text))
    if test["kind"] == "assertion":
        assertions = re.findall(r"Assertion failed: ([^\r\n]+)", text)
        passed = (
            bool(assertions)
            and set(assertions) == {test["assertion"]}
            and "Fatal:" in text
        )
        passed = passed and not any(int(value) for value in severities.values())
        if not passed:
            raise VerificationFailure("expected_assertion", test["name"], logfile)
        return {"status": "pass", "checks": 1}
    if process.returncode:
        raise VerificationFailure("simulation", test["name"], logfile)
    if any(int(value) for value in severities.values()):
        raise VerificationFailure("checking", test["name"], logfile)
    if not result_file.is_file():
        raise VerificationFailure("result_missing", test["name"], logfile)
    data = json.loads(result_file.read_text())
    required = {
        "status",
        "comparisons",
        "checks",
        "expected",
        "received",
        "cancelled",
        "pending",
    }
    if set(data) != required:
        raise ValueError(f"Invalid verification evidence: {result_file}")
    passed = data["status"] == "pass" and data["comparisons"] + data["checks"] > 0
    passed = (
        passed
        and data["pending"] == 0
        and data["expected"] == data["comparisons"] + data["cancelled"]
    )
    passed = (
        passed
        and data["received"] == data["comparisons"]
        and severities == {"ERROR": "0", "FATAL": "0"}
    )
    passed = passed and not re.search(
        r"^(?:Error|Fatal):|^UVM_(?:ERROR|FATAL)\s+(?!:)", text, re.MULTILINE
    )
    if not passed:
        raise VerificationFailure("checking", test["name"], logfile)
    return data


def run_profile(repo, design, profile, directory, do_run, ctx):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "configuration.json").write_text(json.dumps(profile, indent=2) + "\n")
    artifacts = build(repo, design, profile, directory, ctx)
    output = artifacts["output"]
    records = []
    if not do_run:
        return {
            "ip": profile["ip"],
            "profile": profile["profile"],
            "status": "built",
            "output": str(output),
        }
    for test in profile["tests"]:
        record_dir = directory / test["name"]
        record_dir.mkdir(parents=True, exist_ok=True)
        evidence = record_dir / "evidence.json"
        if evidence.exists():
            raise FileExistsError(evidence)
        binary = artifacts["binaries"][test["name"]]
        if test["metrics"]:
            shutil.copytree(Path(str(binary) + ".vdb"), record_dir / "simulation.vdb")
        arguments = [
            "uvm-simulate",
            binary,
            "-no_save",
            f"+UVM_TESTNAME={test['test']}",
            f"+RESULT={evidence}",
            f"+TIMEOUT_NS={test['timeout_ns']}",
            f"+SINK_READY={test['sink_ready']}",
            f"+SINK_STALL={test['sink_stall']}",
            *test["plusargs"],
        ]
        if test["metrics"]:
            arguments.extend(
                [
                    "-cm",
                    "+".join([*test["metrics"], "assert"]),
                    "-cm_dir",
                    record_dir / "simulation.vdb",
                ]
            )
        if profile["model"]["manifest"]:
            arguments.extend(
                [
                    "-sv_lib",
                    Path(repo)
                    / profile["root"]
                    / "build/verification/reference/debug"
                    / profile["model"]["library"],
                ]
            )
        process = execute(
            repo,
            arguments,
            binary.parent,
            record_dir / "simulation.log",
            test["wall_timeout_s"],
            ctx,
        )
        verdict = judge(test, process, evidence)
        record = {
            "ip": profile["ip"],
            "profile": profile["profile"],
            "test": test["name"],
            "id": f"{profile['scope']}/{profile['ip']}/{profile['profile']}/{test['name']}",
            "kind": test["kind"],
            "configuration": profile["config_path"],
            "log": str(record_dir / "simulation.log"),
            **verdict,
            "coverage": None,
        }
        if (
            verdict["status"] == "pass"
            and test["kind"] == "functional"
            and test["metrics"]
        ):
            # Runtime databases belong to this run, never the reusable build directory.
            record["coverage"] = collect(
                repo, test, record_dir / "simulation.vdb", record_dir / "coverage", ctx
            )
        (record_dir / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
    return {
        "ip": profile["ip"],
        "profile": profile["profile"],
        "configuration": profile["config_path"],
        "status": "pass",
        "tests": records,
    }


def run_verification(repo, design, ip, ctx, do_run):
    selected = profiles(repo, design, ip)
    return run_suite(repo, design, selected, ip, ctx, do_run)


def run_suite(repo, design, selected, name, ctx, do_run):
    scope = design if design is not None else selected[0]["scope"]
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    directory = Path(
        log_dir(
            repo, scope, "verilog", stamp, "uvm", name if name is not None else "suite"
        )
    )
    directory.mkdir(parents=True)
    records = []
    for profile in selected:
        records.append(
            run_profile(
                repo,
                design,
                profile,
                directory / profile["ip"] / profile["profile"],
                do_run,
                ctx,
            )
        )
    planned = [
        f"{profile['scope']}/{profile['ip']}/{profile['profile']}/{test['name']}"
        for profile in selected
        for test in profile["tests"]
    ]
    report = {
        "design": design,
        "operation": "run" if do_run else "build",
        "planned_tests": planned,
        "status": "pass",
        "profiles": records,
        "log": str(directory),
    }
    path = directory / "result.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return {
        "design": design,
        "success": True,
        "results": records,
        "failures": [],
        "report_path": str(path),
        "log": str(directory),
    }
