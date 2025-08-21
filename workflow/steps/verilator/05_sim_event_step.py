import os
import subprocess
import sys

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if utils_path not in sys.path:
  sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path

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
  log_dir = f"{arch_dir}/log"
  
  os.makedirs(log_dir, exist_ok=True)
  
  bin_path = f"{build_dir}/obj_dir/VTestHarness"
  binary = data.get("binary", "")
  
  args = ""
  if binary:
    args = f"+permissive +loadmem={binary} +loadmem_addr=80000000 +custom_boot_pin=1 +permissive-off {binary} "
  
  subprocess.run(f"{bin_path} {args}", shell=True, check=True)
  
  await context.emit({"topic": "verilator.complete", "data": {**data, "task": "simulation"}})