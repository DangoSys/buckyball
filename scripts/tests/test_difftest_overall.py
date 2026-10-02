import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import json
import subprocess

spec = importlib.util.spec_from_file_location(
    "overall", Path(__file__).parents[1] / "run_difftest_overall.py"
)
overall = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overall)


class StatisticsTest(unittest.TestCase):
    def test_medians_negative_overhead_and_figures(self):
        rows = []
        for mode, times in [
            ("none", [2, 1, 3]),
            ("access", [4, 100, 3]),
            ("difftest-n", [1, 2, 0.5]),
        ]:
            for run_id, seconds in enumerate(times):
                rows.append(
                    dict(
                        workload="TEST FIXTURE",
                        mode=mode,
                        run_id=run_id,
                        time_s=seconds,
                        cycles=1000,
                        throughput=1000 / seconds,
                        overhead_pct="",
                        verification_bytes=0 if mode == "none" else 760,
                        verification_events=0 if mode == "none" else 10,
                        comparison_time_s=0.01,
                    )
                )
        summary = overall.summarize(rows)
        self.assertEqual([r["overhead_pct"] for r in summary], [0, 100, -50])
        self.assertEqual(summary[1]["time_s"], 4)
        with tempfile.TemporaryDirectory(prefix="overall-test-fixture-") as output:
            overall.write_csv(Path(output) / "summary.csv", summary)
            overall.figures(Path(output), summary)
            self.assertGreater(
                (Path(output) / "figure1_normalized_time.png").stat().st_size, 1000
            )
            self.assertGreater(
                (Path(output) / "figure2_verification_bytes.png").stat().st_size, 1000
            )

    def test_invalid_measurements(self):
        metrics = dict(
            status="PASS",
            mode="access",
            time_s=2.0,
            cycles=20,
            throughput=10.0,
            verification_bytes=100,
            verification_events=1,
            comparison_time_s=0.1,
            verification_data_bytes=72,
            verification_control_bytes=28,
            hardware_verification_events=1,
            reference_events=1,
            matched_events=1,
        )
        overall.validate_metrics(metrics, "access")
        for key, value in [
            ("status", "FAIL"),
            ("cycles", 0),
            ("throughput", 99),
            ("time_s", float("nan")),
            ("verification_events", 0),
            ("verification_bytes", 72),
            ("matched_events", 0),
            ("reference_events", 2),
            ("hardware_verification_events", 0),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                overall.validate_metrics(dict(metrics, **{key: value}), "access")
        with self.assertRaises(ValueError):
            overall.validate_metrics(dict(metrics, mode="none"), "none")

    def test_no_incomplete_summary(self):
        with self.assertRaises(ValueError):
            overall.summarize([dict(workload="TEST FIXTURE", mode="none", time_s=1)])

    def test_runner_rotates_modes_and_stops_on_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory(
                prefix="overall-TEST-FIXTURE-"
            ) as directory:
                root = Path(directory)
                (root / "image").write_bytes(b"TEST FIXTURE, not an executable")
                manifest = [
                    'root="."',
                    'fpga_location="TEST FIXTURE"',
                    "[tool_versions]",
                    'fixture="test only"',
                ]
                for mode in overall.MODES:
                    case = root / mode
                    (case / "fpgaCompDir").mkdir(parents=True)
                    (case / "host").write_bytes(b"TEST FIXTURE")
                    (case / "fpgaCompDir/bitstream.bit").write_bytes(b"TEST FIXTURE")
                    for name, value in [
                        ("p2e_trace_mode", mode),
                        ("dut-config", "TEST FIXTURE"),
                        ("dut-config.sha256", "fixture hash"),
                    ]:
                        (case / name).write_text(value)
                    manifest += [
                        f"[modes.{mode}]",
                        f'host="{mode}/host"',
                        f'bitstream="{mode}/fpgaCompDir/bitstream.bit"',
                    ]
                manifest += [
                    "[[workloads]]",
                    'name="TEST FIXTURE"',
                    'image="image"',
                    'elf="image"',
                    "inputs=[]",
                    "seed=0",
                ]
                (root / "manifest.toml").write_text("\n".join(manifest))
                args = SimpleNamespace(
                    manifest=root / "manifest.toml",
                    output=root / "out",
                    runs=3,
                    timeout=1,
                )
                order = []

                def execute(command, **kwargs):
                    mode = command[command.index("--verification-mode") + 1]
                    order.append(mode)
                    log = Path(command[command.index("--log-dir") + 1])
                    count = 0 if mode == "none" else 10
                    data = count * {"none": 0, "access": 72, "difftest-n": 24}[mode]
                    control = {"none": 0, "access": 28, "difftest-n": 20}[mode]
                    seconds = {"none": 1.0, "access": 2.0, "difftest-n": 1.5}[mode]
                    metrics = dict(
                        status="FAIL" if fail and mode == "access" else "PASS",
                        mode=mode,
                        dut_config="TEST FIXTURE",
                        dut_config_sha256="fixture hash",
                        time_s=seconds,
                        cycles=100,
                        throughput=100 / seconds,
                        comparison_time_s=0.0,
                        verification_events=count,
                        verification_data_bytes=data,
                        verification_control_bytes=control,
                        verification_bytes=data + control,
                        hardware_verification_events=count,
                        reference_events=count,
                        matched_events=count,
                    )
                    (log / "metrics.json").write_text(json.dumps(metrics))
                    return subprocess.CompletedProcess(command, 0)

                with patch.object(
                    overall.subprocess, "run", side_effect=execute
                ), patch.object(
                    overall.subprocess,
                    "check_output",
                    side_effect=lambda *a, **k: (
                        "TEST FIXTURE" if k.get("text") else b"TEST FIXTURE"
                    ),
                ), patch.object(
                    overall, "figures"
                ), patch(
                    "builtins.print"
                ):
                    if fail:
                        with self.assertRaises(ValueError):
                            overall.run(args)
                        self.assertEqual(order, ["none", "access"])
                        self.assertFalse((args.output / "summary.csv").exists())
                        self.assertTrue((args.output / "runs.csv").exists())
                    else:
                        overall.run(args)
                        self.assertEqual(
                            order,
                            [
                                "none",
                                "access",
                                "difftest-n",
                                "access",
                                "difftest-n",
                                "none",
                                "difftest-n",
                                "none",
                                "access",
                            ],
                        )
                        self.assertEqual(
                            len((args.output / "runs.csv").read_text().splitlines()), 10
                        )


if __name__ == "__main__":
    unittest.main()
