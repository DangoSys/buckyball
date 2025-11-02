"""
Sardine test framework configuration.
Provides fixtures and test setup/teardown.
"""

import subprocess
import pytest
import logging
from pathlib import Path
import os
from datetime import datetime
import select
import threading
import time
import re


class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


@pytest.fixture
def script_runner():
    """Execute script and return result."""

    def _run_script(script_path, args=None, timeout=60):
        """Run script and capture output."""
        cmd = [script_path]
        if args:
            cmd.extend(args)

        logger = logging.getLogger()
        logger.info(f"Starting command: {' '.join(cmd)}")

        try:
            # Use Popen with non-blocking mode
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                # No buffering
                bufsize=0,
                universal_newlines=True,
            )

            # Set non-blocking mode
            import fcntl

            fcntl.fcntl(process.stdout.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)
            fcntl.fcntl(process.stderr.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)

            stdout_lines = []
            stderr_lines = []

            start_time = time.time()

            # Read output in real-time
            while True:
                # Check if process has ended
                if process.poll() is not None:
                    break

                # Check timeout
                if timeout and timeout > 0 and (time.time() - start_time) > timeout:
                    process.kill()
                    logger.error(f"Process killed due to timeout ({timeout}s)")
                    break

                # Use select to check if data is ready to read
                ready, _, _ = select.select(
                    [process.stdout, process.stderr], [], [], 0.1
                )

                for stream in ready:
                    try:
                        if stream == process.stdout:
                            data = stream.read()
                            if data:
                                for line in data.splitlines():
                                    if line.strip():
                                        stdout_lines.append(line)
                                        logger.info(f"STDOUT: {line}")
                        elif stream == process.stderr:
                            data = stream.read()
                            if data:
                                for line in data.splitlines():
                                    if line.strip():
                                        stderr_lines.append(line)
                                        logger.warning(f"STDERR: {line}")
                    except Exception:
                        # Non-blocking read may throw exceptions, ignore
                        pass

            # Wait for process to finish and read remaining output
            remaining_stdout, remaining_stderr = process.communicate()
            if remaining_stdout:
                for line in remaining_stdout.splitlines():
                    if line.strip():
                        stdout_lines.append(line)
                        logger.info(f"STDOUT: {line}")
            if remaining_stderr:
                for line in remaining_stderr.splitlines():
                    if line.strip():
                        stderr_lines.append(line)
                        logger.warning(f"STDERR: {line}")

            return {
                "returncode": process.returncode,
                "stdout": "\n".join(stdout_lines),
                "stderr": "\n".join(stderr_lines),
            }

        except Exception as e:
            logger.error(f"Script execution failed: {str(e)}")
            return {"returncode": -1, "stdout": "", "stderr": str(e)}

    return _run_script


@pytest.fixture
def command_run():
    """Execute shell command with real-time output and early termination detection."""

    def _run_command(command, early_exit_pattern=None, timeout=None):
        """
        Run shell command with real-time output.

        Args:
            command: Shell command to execute
            timeout: Optional timeout in seconds (None for no timeout)
            early_exit_pattern: Optional regex pattern to detect early completion

        Returns:
            Dict with returncode, stdout, stderr
        """
        logger = logging.getLogger()
        logger.info(f"Running command: {command}")

        try:
            # Execute command with shell=True
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
                universal_newlines=True,
                executable="bash",
            )

            # Set non-blocking mode
            import fcntl

            fcntl.fcntl(process.stdout.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)
            fcntl.fcntl(process.stderr.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)

            stdout_lines = []
            stderr_lines = []
            start_time = time.time()
            last_output_time = start_time

            # Read output in real-time
            while True:
                current_time = time.time()

                # Check timeout
                if timeout and timeout > 0 and (current_time - start_time) > timeout:
                    logger.error(f"Process killed due to timeout ({timeout}s)")
                    process.kill()
                    break

                # Use select to check if data is ready to read
                ready, _, _ = select.select(
                    [process.stdout, process.stderr], [], [], 0.1
                )

                if ready:
                    last_output_time = current_time
                else:
                    # If no output for 5 seconds, check if process has ended
                    if (current_time - last_output_time) > 5:
                        if process.poll() is not None:
                            logger.info("Process finished naturally")
                            break

                for stream in ready:
                    try:
                        if stream == process.stdout:
                            data = stream.read()
                            if data:
                                for line in data.splitlines():
                                    if line.strip():
                                        line_text = line.strip()
                                        # Output to console in real-time
                                        print(line_text)
                                        stdout_lines.append(line_text)
                                        logger.info(f"STDOUT: {line_text}")

                                        # Detect early exit pattern
                                        if early_exit_pattern and re.search(
                                            early_exit_pattern, line_text
                                        ):
                                            logger.info(
                                                "Detected early exit pattern, terminating process"
                                            )
                                            process.terminate()
                                            # Wait for process to finish
                                            try:
                                                process.wait(timeout=5)
                                            except subprocess.TimeoutExpired:
                                                process.kill()
                                            return {
                                                "returncode": 0,
                                                "stdout": "\n".join(stdout_lines),
                                                "stderr": "\n".join(stderr_lines),
                                            }
                        elif stream == process.stderr:
                            data = stream.read()
                            if data:
                                for line in data.splitlines():
                                    if line.strip():
                                        line_text = line.strip()
                                        stderr_lines.append(line_text)
                                        logger.warning(f"STDERR: {line_text}")
                    except Exception:
                        # Non-blocking read may throw exceptions, ignore
                        pass

            # Wait for process to finish and read remaining output
            try:
                remaining_stdout, remaining_stderr = process.communicate(timeout=5)
                if remaining_stdout:
                    for line in remaining_stdout.splitlines():
                        if line.strip():
                            stdout_lines.append(line)
                            logger.info(f"STDOUT: {line}")
                if remaining_stderr:
                    for line in remaining_stderr.splitlines():
                        if line.strip():
                            stderr_lines.append(line)
                            logger.warning(f"STDERR: {line}")
            except subprocess.TimeoutExpired:
                process.kill()
                logger.error("Process killed after timeout waiting for final output")

            return {
                "returncode": process.returncode,
                "stdout": "\n".join(stdout_lines),
                "stderr": "\n".join(stderr_lines),
            }

        except Exception as e:
            logger.error(f"Command execution failed: {str(e)}")
            return {"returncode": -1, "stdout": "", "stderr": str(e)}

    return _run_command


def pytest_runtest_setup(item):
    """Before each test runs."""
    logger = logging.getLogger()
    # Get current timestamp and process ID
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    pid = os.getpid()
    # Get test function name
    test_func = item.name

    # Get test markers for creating folder name
    markers = [mark.name for mark in item.iter_markers()]
    # Use first marker as folder name, use "default" if no markers
    log_folder_name = markers[0] if markers else "default"

    # Create dedicated logs folder using marker name
    log_dir = Path(__file__).parent / "reports" / f"{timestamp}-{log_folder_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{timestamp}_pid{pid}_{test_func}.log"

    # Create FlushFileHandler
    file_handler = FlushFileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    # Store in item for teardown removal
    item._sardine_log_handler = file_handler
    logger.addHandler(file_handler)
    logger.info(f"=== Starting test: {item.name} (PID: {pid}) ===")
    logger.info(f"Test markers: {markers}")
    logger.info(f"Log file: {log_file}")


def pytest_runtest_teardown(item, nextitem):
    """After each test completes."""
    logger = logging.getLogger()
    logger.info(f"=== Completed test: {item.name} ===")
    # Remove and close handler
    if hasattr(item, "_sardine_log_handler"):
        logger.removeHandler(item._sardine_log_handler)
        item._sardine_log_handler.close()
        del item._sardine_log_handler


def pytest_sessionstart(session):
    """Before test session starts."""
    # Ensure reports and logs directories exist
    reports_dir = Path(__file__).parent / "reports"
    logs_dir = reports_dir / "logs"
    reports_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info("=== Sardine Test Framework Starting ===")
    logger.info(f"Working directory: {Path.cwd()}")
    logger.info(f"Reports will be saved to: {reports_dir}")
    logger.info(f"Individual test logs will be saved to: {logs_dir}/[marker_name]/")


def pytest_sessionfinish(session, exitstatus):
    """After test session completes."""
    reports_dir = Path(__file__).parent / "reports"
    logs_dir = reports_dir / "logs"
    logger = logging.getLogger(__name__)
    logger.info(f"Test reports saved to: {reports_dir}")
    logger.info(f"Individual test logs saved to: {logs_dir}/[marker_name]/")
    logger.info(f"=== Sardine Test Framework Finished (exit: {exitstatus}) ===")
