#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path

from bemu_analysis import abs_log_dir, analysis_dir, chip_maps

BBDIR = Path(__file__).resolve().parents[6]


def _write_bdb(text: str) -> Path:
    tmp = Path(tempfile.mkdtemp())
    (tmp / "bdb.ndjson").write_text(text, encoding="utf-8")
    return tmp


class AbsLogDirTest(unittest.TestCase):
    def test_relative_fails(self):
        with self.assertRaisesRegex(ValueError, "absolute path"):
            abs_log_dir("log/2026-08-19-13-00-bemu-buddy-buckyball-yolo26-run")

    def test_absolute_ok(self):
        path = abs_log_dir("/tmp/bemu-log")
        self.assertTrue(path.is_absolute())


class ChipMapsTest(unittest.TestCase):
    def test_pebble_loads_ballisa(self):
        names, matrix, depth = chip_maps(str(BBDIR), "pebble")
        self.assertEqual(names[0], "fence")
        self.assertEqual(names[32], "mset")
        self.assertEqual(names[65], "smatmul")
        self.assertEqual(names[48], "im2col")
        self.assertEqual(names[49], "transpose")
        self.assertEqual(names[51], "fp2int")
        self.assertEqual(names[52], "int2fp")
        self.assertEqual(matrix, {65})
        self.assertEqual(depth, 1024)


class AnalysisDirTest(unittest.TestCase):
    def setUp(self):
        self.names, self.matrix, self.depth = chip_maps(str(BBDIR), "pebble")

    def test_itrace_mtrace(self):
        log_dir = _write_bdb(
            "\n".join(
                [
                    '{"type":"itrace","clk":10,"event":"complete","funct":"0x00","pc":"0x0","rs1":"0x0","rs2":"0x0"}',
                    '{"type":"itrace","clk":20,"event":"complete","funct":"0x41","pc":"0x0","rs1":"0x0","rs2":"0x0000000100010400"}',
                    '{"type":"mtrace","clk":15,"event":"read","addr":"0x0","rows":1024,"line_bytes":16,"row_stride":16,"vbank_id":0,"pbank_id":0,"group_id":0}',
                    '{"type":"mtrace","clk":16,"event":"write","addr":"0x0","rows":1,"line_bytes":16,"row_stride":16,"vbank_id":1,"pbank_id":1,"group_id":0}',
                    "",
                ]
            )
        )
        text = analysis_dir(
            log_dir,
            names=self.names,
            matrix=self.matrix,
            bank_depth=self.depth,
            itrace=True,
            mtrace=True,
        )
        self.assertIn("span_cycles: 20", text)
        self.assertIn("smatmul", text)
        self.assertIn("(1024,16,256)  n=1", text)
        self.assertIn(f"bank_depth: {self.depth}", text)
        self.assertIn("mean_rows/bank_depth:", text)
        self.assertNotIn("mean_rows/1024:", text)
        self.assertIn("rows=1: 1 (50.0%)", text)

    def test_bank_depth_from_arg(self):
        log_dir = _write_bdb(
            '{"type":"mtrace","clk":1,"event":"read","addr":"0x0","rows":32,"line_bytes":16,"row_stride":16,"vbank_id":0,"pbank_id":0,"group_id":0}\n'
        )
        text = analysis_dir(
            log_dir,
            names=self.names,
            matrix=self.matrix,
            bank_depth=64,
            itrace=False,
            mtrace=True,
        )
        self.assertIn("bank_depth: 64", text)
        self.assertIn("mean_rows: 32.00", text)
        self.assertIn("mean_rows/bank_depth: 0.5000", text)
        self.assertNotIn("mean_rows/1024", text)

    def test_unknown_funct_fails(self):
        log_dir = _write_bdb(
            '{"type":"itrace","clk":1,"event":"complete","funct":"0x99","pc":"0x0","rs1":"0x0","rs2":"0x0"}\n'
        )
        with self.assertRaisesRegex(ValueError, "unknown funct 0x99"):
            analysis_dir(
                log_dir,
                names=self.names,
                matrix=self.matrix,
                bank_depth=self.depth,
                itrace=True,
                mtrace=False,
            )

    def test_missing_itrace_fails(self):
        log_dir = _write_bdb(
            '{"type":"mtrace","clk":1,"event":"read","addr":"0x0","rows":1,"line_bytes":16,"row_stride":16,"vbank_id":0,"pbank_id":0,"group_id":0}\n'
        )
        with self.assertRaisesRegex(ValueError, "no itrace events"):
            analysis_dir(
                log_dir,
                names=self.names,
                matrix=self.matrix,
                bank_depth=self.depth,
                itrace=True,
                mtrace=False,
            )


if __name__ == "__main__":
    unittest.main()
