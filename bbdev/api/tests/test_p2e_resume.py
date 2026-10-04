import importlib.util
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch


API = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API))
spec = importlib.util.spec_from_file_location(
    "p2e_build_api", API / "steps/bebop/p2e/02_buildbitstream_api.step.py"
)
event = importlib.util.module_from_spec(spec)
spec.loader.exec_module(event)


class P2eResumeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.case = self.root / "bebop/build/toy-test"
        self.routed = self.case / "fpgaCompDir/part_b0_f0/pnrDir"
        self.routed.mkdir(parents=True)
        self.artifacts = [self.routed / name for name in (
            "xepic_vvac_top_0_0_route.dcp", "xepic_vvac_top_0_0.bit"
        )]
        for path in self.artifacts:
            path.write_bytes(b"existing artifact")
        self.context = SimpleNamespace(enqueue=AsyncMock(), trace_id="test")

    async def request(self, **options):
        body = {"chip": "toy", "diff": True, "resume-post-route": True,
                "vsrc_dir": str(self.root / "rtl"), "output_dir": str(self.case)}
        body.update(options)
        with patch.object(event, "get_buckyball_path", return_value=str(self.root)):
            return await event.handler(SimpleNamespace(body=body), self.context)

    async def test_routed_case_is_enqueued_without_mutation(self):
        response = await self.request()
        self.assertEqual(response.status, 202)
        data = self.context.enqueue.await_args.args[0]["data"]
        self.assertTrue(data["resume_post_route"])
        for path in self.artifacts:
            self.assertEqual(path.read_bytes(), b"existing artifact")

    async def test_incomplete_case_is_rejected_before_enqueue(self):
        self.artifacts[1].unlink()
        response = await self.request()
        self.assertEqual(response.status, 400)
        self.context.enqueue.assert_not_awaited()

    async def test_resume_requires_explicit_case(self):
        response = await self.request(output_dir=None)
        self.assertEqual(response.status, 400)
        self.context.enqueue.assert_not_awaited()

    async def test_case_outside_build_root_is_rejected(self):
        response = await self.request(output_dir=str(self.root))
        self.assertEqual(response.status, 400)
        self.context.enqueue.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
