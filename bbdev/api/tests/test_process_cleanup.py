import os
import shlex
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.process import _exists, terminate_group
from utils.process_registry import cancel_process, register_process, unregister_process
from utils.stream_run import stream_run


class ProcessCleanupTest(unittest.TestCase):
    def test_nested_session_exits_on_shutdown_and_cancel(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                parent = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        "import subprocess, sys, time; "
                        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'], start_new_session=True); "
                        "print(child.pid, flush=True); time.sleep(300)",
                    ],
                    stdout=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                child_pid = int(parent.stdout.readline())
                try:
                    if cancel:
                        register_process("nested-session-test", parent.pid)
                        self.assertTrue(cancel_process("nested-session-test"))
                        parent.wait(timeout=2)
                    else:
                        terminate_group(parent, term_timeout=1)
                    self.assertFalse(_exists(child_pid))
                finally:
                    unregister_process("nested-session-test")
                    if _exists(child_pid):
                        os.kill(child_pid, signal.SIGKILL)
                    if parent.poll() is None:
                        parent.kill()
                    parent.wait()
                    parent.stdout.close()

    def test_timeout_exits_nested_session(self):
        with tempfile.TemporaryDirectory() as directory:
            child_pid_file = Path(directory) / "child.pid"
            command = shlex.join(
                [
                    sys.executable,
                    "-c",
                    "import subprocess, sys, time; "
                    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'], start_new_session=True); "
                    "open(sys.argv[1], 'w').write(str(child.pid)); time.sleep(300)",
                    str(child_pid_file),
                ]
            )
            try:
                result = stream_run(command, timeout=1)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(_exists(int(child_pid_file.read_text())))
            finally:
                if child_pid_file.exists():
                    child_pid = int(child_pid_file.read_text())
                    if _exists(child_pid):
                        os.kill(child_pid, signal.SIGKILL)
