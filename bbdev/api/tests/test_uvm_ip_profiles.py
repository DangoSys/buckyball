import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steps.uvm.scripts.uvm_common import load_ip
from steps.uvm.scripts.report import verification_targets


class IpProfiles(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        registry = self.repo / "arch/uvm.toml"
        registry.parent.mkdir(parents=True)
        registry.write_text('ips=["unit"]\n[paths]\nunit="ip"\n')
        self.resources = self.repo / "ip/src/main/resources"
        self.resources.mkdir(parents=True)
        (self.resources / "common.f").write_text("")
        (self.resources / "goban.f").write_text("")
        (self.resources / "goban.toml").write_text('chips=["goban"]\n')

    def tearDown(self):
        self.temp.cleanup()

    def test_build_and_report_choose_same_profile(self):
        for chip, names in [("toy", ["common"]), ("goban", ["common", "goban"])]:
            self.assertEqual([x["name"] for x in load_ip(str(self.repo), "unit", chip=chip)["targets"]], names)
            self.assertEqual(verification_targets(self.repo, chip, ip="unit"), [f"ip/unit/{n}" for n in names])
        with self.assertRaisesRegex(ValueError, "does not support chip"):
            load_ip(str(self.repo), "unit", "goban", "toy")

    def test_profile_and_negative_assertion_can_combine(self):
        (self.resources / "goban.toml").write_text('chips=["goban"]\nexpected_assertion="precise failure"\n')
        target = load_ip(str(self.repo), "unit", "goban", "goban")["targets"][0]
        self.assertEqual(target["expected_assertion"], "precise failure")

    def test_invalid_metadata_and_empty_selection_fail(self):
        for value in ['[]', '[""]', '[" goban"]', '[7]', '"goban"']:
            (self.resources / "goban.toml").write_text(f"chips={value}\n")
            with self.assertRaisesRegex(ValueError, "nonempty list"):
                load_ip(str(self.repo), "unit", chip="goban")
        (self.resources / "goban.toml").write_text('chips=["goban"]\n')
        (self.resources / "common.toml").write_text('chips=["goban"]\n')
        with self.assertRaisesRegex(ValueError, "no applicable targets"):
            load_ip(str(self.repo), "unit", chip="toy")


if __name__ == "__main__":
    unittest.main()
