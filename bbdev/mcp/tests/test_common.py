import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common


class SubmitTest(unittest.TestCase):
    def setUp(self):
        common._submitted_trace_ids.clear()

    @patch("common._http")
    @patch("common._ensure", return_value=5100)
    def test_submit_returns_trace_without_reading_state(self, ensure, http):
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

    @patch("common._read_state", return_value={"unknown": True})
    def test_task_status_rejects_invalid_state(self, read_state):
        with self.assertRaisesRegex(RuntimeError, "invalid task state"):
            common.task_status("trace-1")


if __name__ == "__main__":
    unittest.main()
