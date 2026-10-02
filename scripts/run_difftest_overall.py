#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import signal
import subprocess
import time
import tomllib

MODES = ("none", "access", "difftest-n")
FIELDS = (
    "workload",
    "mode",
    "run_id",
    "time_s",
    "cycles",
    "throughput",
    "overhead_pct",
    "verification_bytes",
    "verification_events",
    "comparison_time_s",
)


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize(rows):
    output = []
    for workload in dict.fromkeys(row["workload"] for row in rows):
        groups = {
            mode: [
                row
                for row in rows
                if row["workload"] == workload and row["mode"] == mode
            ]
            for mode in MODES
        }
        if any(len(group) < 3 for group in groups.values()):
            raise ValueError(
                f"{workload}: each mode requires at least three successful measurements"
            )
        baseline = statistics.median(row["time_s"] for row in groups["none"])
        for mode, group in groups.items():
            result = {
                field: statistics.median(row[field] for row in group)
                for field in FIELDS
                if field not in ("workload", "mode", "run_id", "overhead_pct")
            }
            result.update(
                workload=workload,
                mode=mode,
                run_id="median",
                overhead_pct=100 * (result["time_s"] / baseline - 1),
            )
            output.append(result)
            for row in group:
                row["overhead_pct"] = 100 * (row["time_s"] / baseline - 1)
    return output


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row[field] for field in FIELDS} for row in rows)


def figures(output, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    workloads = list(dict.fromkeys(row["workload"] for row in summary))
    values = {(row["workload"], row["mode"]): row for row in summary}
    for filename, modes, field, ylabel in (
        ("figure1_normalized_time.png", MODES, "time_s", "Normalized execution time"),
        (
            "figure2_verification_bytes.png",
            MODES[1:],
            "verification_bytes",
            "Verification DPI payload (bytes)",
        ),
    ):
        fig, axis = plt.subplots(figsize=(max(6, len(workloads) * 2), 4))
        width = 0.8 / len(modes)
        for i, mode in enumerate(modes):
            heights = [
                values[(w, mode)][field]
                / (values[(w, "none")][field] if field == "time_s" else 1)
                for w in workloads
            ]
            positions = [
                j + (i - (len(modes) - 1) / 2) * width for j in range(len(workloads))
            ]
            axis.bar(positions, heights, width, label=mode)
        axis.set_xticks(range(len(workloads)), workloads)
        axis.set_ylabel(ylabel)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)


def validate_metrics(metrics, mode):
    if metrics["status"] != "PASS" or metrics["mode"] != mode:
        raise ValueError("run did not pass in the requested verification mode")
    for field in ("time_s", "cycles", "throughput"):
        if not math.isfinite(metrics[field]) or metrics[field] <= 0:
            raise ValueError(f"invalid {field}")
    if not math.isclose(
        metrics["throughput"], metrics["cycles"] / metrics["time_s"], rel_tol=1e-9
    ):
        raise ValueError("throughput does not match measured cycles/time")
    for field in ("verification_events", "verification_bytes", "comparison_time_s"):
        if not math.isfinite(metrics[field]) or metrics[field] < 0:
            raise ValueError(f"invalid {field}")
    data_bytes = (
        metrics["verification_events"]
        * {"none": 0, "access": 72, "difftest-n": 24}[mode]
    )
    if (
        metrics["verification_data_bytes"] != data_bytes
        or metrics["verification_bytes"]
        != data_bytes + metrics["verification_control_bytes"]
    ):
        raise ValueError("verification DPI payload accounting mismatch")
    if any(
        metrics[field] != metrics["verification_events"]
        for field in (
            "hardware_verification_events",
            "reference_events",
            "matched_events",
        )
    ):
        raise ValueError("hardware/host/reference/matched counts differ")
    if mode == "none":
        if metrics["verification_events"] != 0 or metrics["verification_bytes"] != 0:
            raise ValueError("No Verification generated verification traffic")
    elif metrics["verification_events"] == 0:
        raise ValueError("verification run checked no events")


def build(args):
    path = args.manifest.resolve()
    manifest = tomllib.loads(path.read_text())
    root = (path.parent / manifest["root"]).resolve()
    cargo_manifest = root / manifest["build"]["cargo_manifest"]
    for mode in MODES:
        case = (root / manifest["modes"][mode]["bitstream"]).parent.parent
        if case.exists():
            raise FileExistsError(f"build requires a fresh output directory: {case}")
        case.mkdir(parents=True)
        rtl = root / "arch/build/p2e-overall" / mode
        if rtl.exists():
            raise FileExistsError(f"build requires a fresh RTL directory: {rtl}")
        env = dict(
            os.environ,
            VSRC_PATH=str(rtl),
            OUT_PATH=str(case),
            P2E_VERIFICATION_MODE=mode,
            CARGO_TARGET_DIR=str(root / "bebop/target/p2e-overall" / mode),
        )
        features = "p2e" if mode == "none" else "p2e,bemu"
        commands = [
            (
                [
                    "mill",
                    "-i",
                    "buckyball.runMain",
                    "sims.p2e.Elaborate",
                    manifest["build"]["rtl_config"],
                    f"--verification-mode={mode}",
                    f"-o={rtl}",
                ],
                root / "arch",
            ),
            (
                [
                    "cargo",
                    "run",
                    "--release",
                    "--manifest-path",
                    str(cargo_manifest),
                    "--features",
                    features,
                    "--",
                    "build",
                    "p2e",
                    "--verification-mode",
                    mode,
                    "--rtl-dir",
                    str(rtl),
                    "--out-dir",
                    str(case),
                ],
                root / "bebop",
            ),
        ]
        started = time.monotonic()
        with (case.parent / f"{mode}-build.log").open("w") as log:
            for command, cwd in commands:
                subprocess.run(
                    command,
                    cwd=cwd,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        (case / "build-metrics.json").write_text(
            json.dumps(
                {
                    "build_time_s": time.monotonic() - started,
                    "commands": [item[0] for item in commands],
                },
                indent=2,
            )
            + "\n"
        )


def run(args):
    manifest_path = args.manifest.resolve()
    manifest = tomllib.loads(manifest_path.read_text())
    if args.output is None:
        raise ValueError("--output is required for measurements")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.runs < 3:
        raise ValueError("--runs must be at least 3")
    root = (manifest_path.parent / manifest["root"]).resolve()
    artifacts = manifest["modes"]
    if set(artifacts) != set(MODES):
        raise ValueError("manifest must specify exactly none/access/difftest-n")
    files = {}
    configs = []
    for mode in MODES:
        artifact = artifacts[mode]
        for field in ("host", "bitstream"):
            artifact[field] = str((root / artifact[field]).resolve(strict=True))
            files[artifact[field]] = digest(artifact[field])
        case = Path(artifact["bitstream"]).parent.parent
        if (case / "p2e_trace_mode").read_text() != mode:
            raise ValueError(f"{mode}: bitstream mode mismatch")
        configs.append(
            (
                (case / "dut-config").read_text(),
                (case / "dut-config.sha256").read_text(),
            )
        )
    if len(set(configs)) != 1:
        raise ValueError("DUT configuration differs across modes")
    for workload in manifest["workloads"]:
        for field in ("image", "elf"):
            workload[field] = str((root / workload[field]).resolve(strict=True))
            files[workload[field]] = digest(workload[field])
        for path in workload["inputs"]:
            path = (root / path).resolve(strict=True)
            files[str(path)] = digest(path)
        if not isinstance(workload["seed"], int):
            raise ValueError("each workload must declare its fixed input seed")
    metadata = {
        "manifest": manifest,
        "files_sha256": files,
        "dut_config": configs[0],
        "host_platform": platform.platform(),
        "python": platform.python_version(),
        "tool_versions": manifest["tool_versions"],
        "tool_versions_source": "operator supplied manifest",
        "runs_per_mode": args.runs,
        "affinity": sorted(os.sched_getaffinity(0)),
        "measurement": "P2E hardware",
        "git": {
            name: subprocess.check_output(
                ["git", "-C", str(root / name), "rev-parse", "HEAD"], text=True
            ).strip()
            for name in (".", "bebop")
        },
    }
    metadata["source_files_sha256"] = {}
    for repo, paths in (
        (root, ["arch/src", "examples", "scripts", "bb-tests/workloads/src/CTest"]),
        (root / "bebop", ["src", "tests", "Cargo.toml", "build.rs"]),
    ):
        names = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                *paths,
            ]
        )
        for name in names.decode().split("\0"):
            source = repo / name
            if name and source.is_file():
                metadata["source_files_sha256"][str(source.relative_to(root))] = digest(
                    source
                )
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "unavailable.json").write_text(
        json.dumps(manifest.get("unavailable", []), indent=2) + "\n"
    )
    for name in (".", "bebop"):
        patch = subprocess.check_output(
            ["git", "-C", str(root / name), "diff", "HEAD", "--binary"]
        )
        (output / ("root.patch" if name == "." else "bebop.patch")).write_bytes(patch)
    rows = []
    for workload in manifest["workloads"]:
        for run_id in range(
            1, (1 if workload.get("validation_only", False) else args.runs) + 1
        ):
            order = MODES[(run_id - 1) % 3 :] + MODES[: (run_id - 1) % 3]
            for mode in order:
                for path in files:
                    if digest(path) != files[path]:
                        raise ValueError(f"artifact changed during experiment: {path}")
                log_dir = output / workload["name"] / mode / str(run_id)
                log_dir.mkdir(parents=True)
                command = [
                    artifacts[mode]["host"],
                    "run",
                    "p2e",
                    "--verification-mode",
                    mode,
                    "--image",
                    workload["image"],
                    "--image-elf",
                    workload["elf"],
                    "--bitstream",
                    artifacts[mode]["bitstream"],
                    "--log-dir",
                    str(log_dir),
                    "--fpga-location",
                    manifest["fpga_location"],
                ]
                (log_dir / "command.json").write_text(json.dumps(command) + "\n")
                started = time.monotonic()
                process = {"status": "FAIL"}
                try:
                    with (log_dir / "host.log").open("w") as log:
                        result = subprocess.run(
                            command,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=args.timeout,
                        )
                    process["exit_code"] = result.returncode
                    result.check_returncode()
                    process["status"] = "PASS"
                except subprocess.TimeoutExpired:
                    process["reason"] = "host execution timed out"
                    pid_file = log_dir / "vdbg.pid"
                    if pid_file.exists():
                        os.killpg(int(pid_file.read_text()), signal.SIGKILL)
                    raise
                finally:
                    process["wall_time_s"] = time.monotonic() - started
                    (log_dir / "process.json").write_text(json.dumps(process) + "\n")
                metrics = json.loads((log_dir / "metrics.json").read_text())
                validate_metrics(metrics, mode)
                if (metrics["dut_config"], metrics["dut_config_sha256"]) != configs[0]:
                    raise ValueError("runtime DUT configuration mismatch")
                row = {
                    field: metrics[field]
                    for field in FIELDS
                    if field not in ("workload", "mode", "run_id", "overhead_pct")
                }
                row.update(
                    workload=workload["name"], mode=mode, run_id=run_id, overhead_pct=""
                )
                if not workload.get("validation_only", False):
                    rows.append(row)
                    write_csv(output / "runs.csv", rows)
    summary = summarize(rows)
    write_csv(output / "runs.csv", rows)
    write_csv(output / "summary.csv", summary)
    figures(output, summary)
    print("workload, mode, median seconds, cycles/s, overhead %, verification bytes")
    for row in summary:
        print(
            f"{row['workload']}, {row['mode']}, {row['time_s']:.6f}, {row['throughput']:.2f}, "
            f"{row['overhead_pct']:.2f}, {row['verification_bytes']:.0f}"
        )
    for workload in dict.fromkeys(row["workload"] for row in summary):
        times = {
            row["mode"]: row["time_s"] for row in summary if row["workload"] == workload
        }
        print(
            f"{workload}: DiffTest-N vs access speedup={times['access'] / times['difftest-n']:.3f}x; "
            f"remaining overhead={100 * (times['difftest-n'] / times['none'] - 1):.2f}%"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run three P2E verification configurations and retain raw measurements"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="build all three P2E artifacts; do not run experiments",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    if args.build_only:
        build(args)
    else:
        run(args)
