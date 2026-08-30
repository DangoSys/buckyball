import unittest
from types import SimpleNamespace

from compiler.scripts.pb_to_target_registry import _rushb_targets


class RushBTargetsTest(unittest.TestCase):
    def test_encodes_tile_and_local_core_indices(self):
        mapped = SimpleNamespace(
            role="pebble",
            pkg="",
            balldomain=SimpleNamespace(mappings=[object()]),
        )
        chip = SimpleNamespace(
            cores=[mapped, mapped, mapped],
            tiles=[
                SimpleNamespace(core_indices=[0, 1]),
                SimpleNamespace(core_indices=[2]),
            ],
        )

        self.assertEqual(
            _rushb_targets(chip),
            [(0, "pebble"), (1, "pebble"), (1 << 16, "pebble")],
        )


if __name__ == "__main__":
    unittest.main()
