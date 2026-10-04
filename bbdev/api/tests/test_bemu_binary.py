import importlib.util
from pathlib import Path
import tempfile
import unittest


API = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "bemu_sim_event", API / "steps/bebop/bemu/02_sim_event.step.py"
)
event = importlib.util.module_from_spec(spec)
spec.loader.exec_module(event)


class BemuBinaryResolutionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def binary(self, relative):
        path = self.root / "bb-tests/output" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return str(path)

    def test_same_model_image_on_two_chips(self):
        for chip in ("toy", "pebble"):
            self.binary(f"kernel/{chip}/fw_payload-alexnet.elf")
        for chip in ("toy", "pebble"):
            expected = self.root / f"bb-tests/output/kernel/{chip}/fw_payload-alexnet.elf"
            self.assertEqual(
                event.resolve_bemu_binary(str(self.root), chip, expected.name),
                str(expected),
            )

    def test_workload_must_belong_to_requested_chip(self):
        path = self.binary("toy/workloads/src/shared-test")
        self.assertEqual(event.resolve_bemu_binary(str(self.root), "toy", "shared-test"), path)
        self.assertIsNone(event.resolve_bemu_binary(str(self.root), "pebble", "shared-test"))

    def test_ambiguous_workload_within_chip_is_an_error(self):
        self.binary("toy/workloads/src/one/shared-test")
        self.binary("toy/workloads/src/two/shared-test")
        with self.assertRaises(ValueError):
            event.resolve_bemu_binary(str(self.root), "toy", "shared-test")


if __name__ == "__main__":
    unittest.main()
