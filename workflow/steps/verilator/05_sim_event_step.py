from contextlib import redirect_stdout
import os
import subprocess
import sys
from datetime import datetime
from unittest import result

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if utils_path not in sys.path:
  sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path
from utils.run import run 

config = {
  "type": "event",
  "name": "make sim",
  "description": "run simulation",
  "subscribes": ["verilator.sim"],
  "emits": ["verilator.complete", "verilator.error"],
  "flows": ["verilator"],
}

async def handler(data, context):
  bbdir = get_buckyball_path()
  arch_dir = f"{bbdir}/arch"
  build_dir = f"{arch_dir}/build"
  
  # 生成时间戳
  timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
  
  # 获取binary名字（去除路径）
  binary = data.get("binary", "")
  binary_name = os.path.basename(binary) if binary else "no_binary"
  
  # 创建带时间戳和binary名字的日志目录
  log_dir = f"{arch_dir}/log/{timestamp}-{binary_name}"
  topname = "TestHarness"
  
  os.makedirs(log_dir, exist_ok=True)
  
  bin_path = f"{build_dir}/obj_dir/V{topname}"
  
  args = ""
  if binary:
    args = f"+permissive +loadmem={binary} +loadmem_addr=80000000 +permissive-off {binary} "
  
  # redirect = f"2> >(spike-dasm > {log_dir}/disasm.log)"

  # subprocess.run(f"{bin_path} {args}", shell=True, check=True, text=True)
  # subprocess.run(f"{bin_path} {args}", check=True, text=True)
  subprocess.run(f"source {bbdir}/env.sh && ./scripts/sim.sh {bin_path} {binary} {log_dir}/stdout.log {log_dir}/disasm.log", cwd=os.path.dirname(__file__), shell=True, check=True, text=True)

  await context.emit({"topic": "verilator.complete", "data": {**data, "task": "sim"}})