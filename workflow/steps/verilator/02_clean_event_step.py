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
  "name": "build-clean",
  "description": "clean build directory",
  "subscribes": ["build.start"],
  "emits": ["build.verilog", "build.error"],
  "flows": ["verilator"],
}

async def handler(data, context):
  if not data.get("clean"):
    await context.emit({"topic": "build.verilog", "data": data})
    return
  
  bbdir = get_buckyball_path()
  build_dir = f"{bbdir}/arch/build"
  
  subprocess.run(f"rm -rf {build_dir}", shell=True, check=True)
  
  await context.emit({"topic": "build.verilog", "data": data})