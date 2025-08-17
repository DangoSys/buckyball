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
      # 使用Popen并设置非阻塞
      process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0,  # 无缓冲
        universal_newlines=True
      )
      
      # 设置非阻塞模式
      import fcntl
      fcntl.fcntl(process.stdout.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)
      fcntl.fcntl(process.stderr.fileno(), fcntl.F_SETFL, os.O_NONBLOCK)
      
      stdout_lines = []
      stderr_lines = []
      
      start_time = time.time()
      
      # 实时读取输出
      while True:
        # 检查进程是否结束
        if process.poll() is not None:
          break
          
        # 检查超时
        if timeout and timeout > 0 and (time.time() - start_time) > timeout:
          process.kill()
          logger.error(f"Process killed due to timeout ({timeout}s)")
          break
        
        # 使用select检查是否有数据可读
        ready, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)
        
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
          except:
            # 非阻塞读取可能会抛出异常，忽略
            pass
      
      # 等待进程结束并读取剩余输出
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
        "stderr": "\n".join(stderr_lines)
      }
      
    except Exception as e:
      logger.error(f"Script execution failed: {str(e)}")
      return {
        "returncode": -1,
        "stdout": "",
        "stderr": str(e)
      }
  
  return _run_script


def pytest_runtest_setup(item):
  """Before each test runs."""
  logger = logging.getLogger()
  # 获取当前时间戳和进程ID
  timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  pid = os.getpid()
  # 获取测试函数名
  test_func = item.name
  
  # 获取测试的marker，用于创建文件夹名
  markers = [mark.name for mark in item.iter_markers()]
  # 使用第一个marker作为文件夹名，如果没有marker则使用"default"
  log_folder_name = markers[0] if markers else "default"
  
  # 创建专门的logs文件夹，使用marker名称
  log_dir = Path(__file__).parent / "reports" / f"{timestamp}-{log_folder_name}"
  log_dir.mkdir(parents=True, exist_ok=True)
  log_file = log_dir / f"{timestamp}_pid{pid}_{test_func}.log"
  
  # 创建FlushFileHandler
  file_handler = FlushFileHandler(log_file, mode="w", encoding="utf-8")
  file_handler.setLevel(logging.DEBUG)
  formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
  file_handler.setFormatter(formatter)
  # 标记到item，方便teardown移除
  item._sardine_log_handler = file_handler
  logger.addHandler(file_handler)
  logger.info(f"=== Starting test: {item.name} (PID: {pid}) ===")
  logger.info(f"Test markers: {markers}")
  logger.info(f"Log file: {log_file}")


def pytest_runtest_teardown(item, nextitem):
  """After each test completes."""
  logger = logging.getLogger()
  logger.info(f"=== Completed test: {item.name} ===")
  # 移除并关闭handler
  if hasattr(item, "_sardine_log_handler"):
    logger.removeHandler(item._sardine_log_handler)
    item._sardine_log_handler.close()
    del item._sardine_log_handler


def pytest_sessionstart(session):
  """Before test session starts."""
  # 确保reports和logs目录存在
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
  