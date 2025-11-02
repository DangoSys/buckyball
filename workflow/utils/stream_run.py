import subprocess
import threading
from typing import Optional, List, Callable


class StreamResult:
    """Result object mimicking subprocess.CompletedProcess"""

    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def stream_run(
    cmd: str,
    cwd: Optional[str] = None,
    shell: bool = True,
    executable: Optional[str] = None,
    timeout: Optional[float] = None,
    on_stdout: Optional[Callable[[str], None]] = None,
    on_stderr: Optional[Callable[[str], None]] = None,
    stdout_prefix: str = "STDOUT",
    stderr_prefix: str = "STDERR",
) -> StreamResult:
    """
    Execute command and stream output in real-time

    Args:
      cmd: Command to execute
      cwd: Working directory
      shell: Whether to execute using shell
      timeout: Timeout in seconds
      on_stdout: Callback function for stdout lines
      on_stderr: Callback function for stderr lines
      stdout_prefix: Prefix for stdout output
      stderr_prefix: Prefix for stderr output

    Returns:
      StreamResult: Result object containing returncode, stdout, stderr

    Example:
      def log_stdout(line):
        logger.info(f'[STDOUT] {line}')

      def log_stderr(line):
        logger.info(f'[STDERR] {line}')

      result = stream_run(
        "make build",
        cwd="/path/to/project",
        on_stdout=log_stdout,
        on_stderr=log_stderr
      )
    """

    def read_stream(
        stream, output_list: List[str], callback: Optional[Callable], prefix: str
    ):
        """Thread function to read stream output"""
        try:
            for line in iter(stream.readline, ""):
                if line:
                    line = line.rstrip()
                    output_list.append(line)
                    if callback:
                        callback(line)
        finally:
            stream.close()

    # Start process
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        executable=executable,
    )

    stdout_lines = []
    stderr_lines = []

    # Create threads to read stdout and stderr
    stdout_thread = threading.Thread(
        target=read_stream,
        args=(process.stdout, stdout_lines, on_stdout, stdout_prefix),
    )
    stderr_thread = threading.Thread(
        target=read_stream,
        args=(process.stderr, stderr_lines, on_stderr, stderr_prefix),
    )

    stdout_thread.start()
    stderr_thread.start()

    try:
        # Wait for process to finish (with timeout)
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Kill process on timeout
        process.kill()
        process.wait()

    # Wait for threads to finish
    stdout_thread.join()
    stderr_thread.join()

    return StreamResult(
        returncode=process.returncode,
        stdout="\n".join(stdout_lines),
        stderr="\n".join(stderr_lines),
    )


def stream_run_logger(
    cmd: str,
    logger,
    cwd: Optional[str] = None,
    shell: bool = True,
    executable: Optional[str] = None,
    timeout: Optional[float] = None,
    stdout_prefix: str = "STDOUT",
    stderr_prefix: str = "STDERR",
    verbose: bool = False,
) -> StreamResult:
    """
    Convenience function for streaming output using logger

    Args:
      cmd: Command to execute
      logger: Logger instance
      cwd: Working directory
      shell: Whether to execute using shell
      timeout: Timeout in seconds
      stdout_prefix: Prefix for stdout output
      stderr_prefix: Prefix for stderr output
      verbose: Whether to use verbose output mode (verbose mode uses logger with timestamp, non-verbose prints directly)

    Returns:
      StreamResult: Result object containing returncode, stdout, stderr
    """

    def log_stdout(line):
        if verbose:
            # Verbose mode: use logger.info, includes timestamp and task ID
            logger.info(f"[{stdout_prefix}] {line}")
        else:
            # Non-verbose mode: print directly, use green for STDOUT
            print(f"\033[32m[{stdout_prefix}]\033[0m {line}")

    def log_stderr(line):
        if verbose:
            # Verbose mode: use logger.info, includes timestamp and task ID
            logger.info(f"[{stderr_prefix}] {line}")
        else:
            # Non-verbose mode: print directly, use red for STDERR
            print(f"\033[31m[{stderr_prefix}]\033[0m {line}")

    return stream_run(
        cmd=cmd,
        cwd=cwd,
        shell=shell,
        executable=executable,
        timeout=timeout,
        on_stdout=log_stdout,
        on_stderr=log_stderr,
        stdout_prefix=stdout_prefix,
        stderr_prefix=stderr_prefix,
    )
