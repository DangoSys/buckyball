import sys
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common


class SubmitTest(unittest.TestCase):
    def setUp(self):
        common._submitted_trace_ids.clear()

    @patch("common._http")
    @patch("common._assert_workspace_workers")
    @patch("common._ensure", return_value=5100)
    def test_submit_returns_trace_without_reading_state(self, ensure, workers, http):
        http.return_value = 202, {"trace_id": "trace-1"}

        result = common.submit("/workload/build", {"chip": "pebble"})

        self.assertEqual(
            result,
            {
                "accepted": True,
                "processing": True,
                "trace_id": "trace-1",
                "port": 5100,
            },
        )
        ensure.assert_called_once_with()
        workers.assert_called_once_with()
        http.assert_called_once_with(
            "POST",
            "http://127.0.0.1:5100/workload/build",
            {"chip": "pebble"},
            timeout=30,
        )

    @patch("common._read_state", return_value=None)
    def test_task_status_reports_queued_before_state_exists(self, read_state):
        common._submitted_trace_ids.add("trace-1")
        self.assertEqual(
            common.task_status("trace-1"),
            {
                "accepted": True,
                "processing": True,
                "queued": True,
                "trace_id": "trace-1",
            },
        )
        read_state.assert_called_once_with("trace-1")

    @patch("common._read_state", return_value=None)
    def test_task_status_rejects_unknown_trace(self, read_state):
        with self.assertRaisesRegex(RuntimeError, "unknown task trace_id"):
            common.task_status("unknown")


class EnsureTest(unittest.TestCase):
    def setUp(self):
        self.proc = common._proc
        self.port = common._port
        self.log_fh = common._log_fh
        common._proc = None
        common._port = None
        common._log_fh = None

    def tearDown(self):
        common._proc = self.proc
        common._port = self.port
        common._log_fh = self.log_fh

    def test_ensure_starts_server_in_nix_environment(self):
        with (
            patch("common.shutil.which", return_value="/usr/bin/nix"),
            patch("common._free_port", return_value=5199),
            patch("common._ready", return_value=True),
            patch("common._assert_workspace_workers"),
            patch("common.open", mock_open()),
            patch("common.subprocess.Popen") as popen,
        ):
            popen.return_value.poll.return_value = None
            self.assertEqual(common._ensure(), 5199)

        popen.assert_called_once_with(
            [
                "nix",
                "develop",
                "--command",
                str(common.BBDEV),
                "start",
                "--server",
                "--port",
                "5199",
            ],
            cwd=str(common.BBDEV.parent),
            stdout=common._log_fh,
            stderr=common._log_fh,
            start_new_session=True,
        )

    @patch("common._read_state")
    def test_task_status_reports_success(self, read_state):
        read_state.return_value = {
            "success": {"body": {"returncode": 0, "task": "build"}}
        }

        self.assertEqual(
            common.task_status("trace-1"),
            {
                "returncode": 0,
                "task": "build",
                "success": True,
                "failure": False,
                "processing": False,
                "trace_id": "trace-1",
            },
        )

    @patch("common._read_state")
    def test_task_status_reports_failure(self, read_state):
        read_state.return_value = {"failure": {"body": {"returncode": 1}}}

        self.assertEqual(
            common.task_status("trace-1"),
            {
                "success": False,
                "failure": True,
                "processing": False,
                "trace_id": "trace-1",
                "body": {"returncode": 1},
                "server_log": str(common.LOG),
            },
        )

    @patch("common._read_state")
    def test_task_status_reports_nested_processing(self, read_state):
        read_state.return_value = {
            "processing": {
                "processing": True,
                "task": "regression.buildbitstream",
                "chip": "pebble",
            }
        }

        self.assertEqual(
            common.task_status("trace-1"),
            {
                "processing": True,
                "task": "regression.buildbitstream",
                "chip": "pebble",
                "accepted": True,
                "trace_id": "trace-1",
            },
        )

    @patch("common._read_state", return_value={"unknown": True})
    def test_task_status_rejects_invalid_state(self, read_state):
        with self.assertRaisesRegex(RuntimeError, "invalid task state"):
            common.task_status("trace-1")


if __name__ == "__main__":
    unittest.main()
