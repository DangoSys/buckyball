import subprocess
import threading
from typing import Optional, List, Callable


class StreamResult:
  """模拟subprocess.CompletedProcess的结果对象"""
  def __init__(self, returncode: int, stdout: str, stderr: str):
    self.returncode = returncode
    self.stdout = stdout
    self.stderr = stderr


def stream_run(
  cmd: str,
  cwd: Optional[str] = None,
  shell: bool = True,
  timeout: Optional[float] = None,
  on_stdout: Optional[Callable[[str], None]] = None,
  on_stderr: Optional[Callable[[str], None]] = None,
  stdout_prefix: str = "STDOUT",
  stderr_prefix: str = "STDERR"
) -> StreamResult:
  """
  执行命令并实时流式输出结果
  
  Args:
    cmd: 要执行的命令
    cwd: 工作目录
    shell: 是否使用shell执行
    timeout: 超时时间（秒）
    on_stdout: stdout行的回调函数
    on_stderr: stderr行的回调函数
    stdout_prefix: stdout输出前缀
    stderr_prefix: stderr输出前缀
  
  Returns:
    StreamResult: 包含returncode, stdout, stderr的结果对象
  
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
  
  def read_stream(stream, output_list: List[str], callback: Optional[Callable], prefix: str):
    """读取流输出的线程函数"""
    try:
      for line in iter(stream.readline, ''):
        if line:
          line = line.rstrip()
          output_list.append(line)
          if callback:
            callback(line)
    finally:
      stream.close()
  
  # 启动进程
  process = subprocess.Popen(
    cmd,
    cwd=cwd,
    shell=shell,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
  )
  
  stdout_lines = []
  stderr_lines = []
  
  # 创建线程来读取stdout和stderr
  stdout_thread = threading.Thread(
    target=read_stream, 
    args=(process.stdout, stdout_lines, on_stdout, stdout_prefix)
  )
  stderr_thread = threading.Thread(
    target=read_stream, 
    args=(process.stderr, stderr_lines, on_stderr, stderr_prefix)
  )
  
  stdout_thread.start()
  stderr_thread.start()
  
  try:
    # 等待进程结束（带超时）
    process.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
    # 超时则终止进程
    process.kill()
    process.wait()
  
  # 等待线程结束
  stdout_thread.join()
  stderr_thread.join()
  
  return StreamResult(
    returncode=process.returncode,
    stdout='\n'.join(stdout_lines),
    stderr='\n'.join(stderr_lines)
  )


def stream_run_logger(
  cmd: str,
  logger,
  cwd: Optional[str] = None,
  shell: bool = True,
  timeout: Optional[float] = None,
  stdout_prefix: str = "STDOUT",
  stderr_prefix: str = "STDERR"
) -> StreamResult:
  """
  使用logger进行流式输出的便捷函数
  
  Args:
    cmd: 要执行的命令
    logger: 日志记录器
    cwd: 工作目录
    shell: 是否使用shell执行
    timeout: 超时时间（秒）
    stdout_prefix: stdout输出前缀
    stderr_prefix: stderr输出前缀
  
  Returns:
    StreamResult: 包含returncode, stdout, stderr的结果对象
  """
  
  def log_stdout(line):
    logger.info(f'[{stdout_prefix}] {line}')
  
  def log_stderr(line):
    logger.info(f'[{stderr_prefix}] {line}')
  
  return stream_run(
    cmd=cmd,
    cwd=cwd,
    shell=shell,
    timeout=timeout,
    on_stdout=log_stdout,
    on_stderr=log_stderr,
    stdout_prefix=stdout_prefix,
    stderr_prefix=stderr_prefix
  )
  