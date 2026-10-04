from pathlib import Path
import tempfile
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steps.bebop.p2e.scripts.runtime_case import validate_runtime_reuse


class RuntimeReuseTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.case = Path(self.directory.name)
        self.bit = self.case / "fpgaCompDir/bitstream.bit"
        for path in (self.bit, self.case / "bebop-p2e",
                     self.case / "vvacDir/runtimeDir/rtcfg",
                     self.case / "vvacDir/runtimeDir/lib/lib_arm/libvCtb.so"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"existing artifact")
        self.mode = self.case / "p2e_trace_mode"
        self.mode.write_text("btrace")
        (self.case / "bebop-p2e").chmod(0o755)

    def test_valid_case_remains_unchanged(self):
        before = {p: p.read_bytes() for p in self.case.rglob("*") if p.is_file()}
        self.assertEqual(validate_runtime_reuse(str(self.bit), True),
                         self.case / "bebop-p2e")
        self.assertEqual(before, {p: p.read_bytes() for p in self.case.rglob("*") if p.is_file()})

    def test_missing_library_is_rejected(self):
        (self.case / "vvacDir/runtimeDir/lib/lib_arm/libvCtb.so").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_runtime_reuse(str(self.bit), True)

    def test_diff_requires_trace_capability(self):
        self.mode.write_text("none")
        with self.assertRaisesRegex(ValueError, "does not support diff"):
            validate_runtime_reuse(str(self.bit), True)
        validate_runtime_reuse(str(self.bit), False)

    def test_unknown_protocol_is_rejected(self):
        self.mode.write_text("unknown")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            validate_runtime_reuse(str(self.bit), False)

    def test_nonexecutable_runtime_is_rejected(self):
        (self.case / "bebop-p2e").chmod(0o644)
        with self.assertRaisesRegex(ValueError, "not executable"):
            validate_runtime_reuse(str(self.bit), True)


if __name__ == "__main__":
    unittest.main()
