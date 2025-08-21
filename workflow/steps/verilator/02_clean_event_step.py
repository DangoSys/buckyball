import subprocess
import os
import sys

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if utils_path not in sys.path:
  sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path

config = {
  "type": "event", 
  "name": "make clean",
  "description": "clean build directory",
  "subscribes": ["verilator.run", "verilator.clean"],
  "emits": ["verilator.verilog", "verilator.complete", "verilator.error"],
  "flows": ["verilator"],
}

async def handler(data, context):
  bbdir = get_buckyball_path()
  build_dir = f"{bbdir}/arch/build"
  subprocess.run(f"rm -rf {build_dir}", shell=True, check=True)
  
  # For run workflow, continue to verilog; for standalone clean, complete
  if data.get("from_run_workflow"):
    await context.emit({"topic": "verilator.verilog", "data": data})
  else:
    await context.emit({"topic": "verilator.complete", "data": {**data, "task": "clean"}})
  