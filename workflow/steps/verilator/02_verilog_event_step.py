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
  "name": "make verilog", 
  "description": "generate verilog code",
  "subscribes": ["verilator.verilog"],
  "emits": ["verilator.build", "verilator.complete", "verilator.error"],
  "flows": ["verilator"],
}

async def handler(data, context):
  bbdir = get_buckyball_path()
  build_dir = f"{bbdir}/arch/build"
  arch_dir = f"{bbdir}/arch"
  
  os.makedirs(build_dir, exist_ok=True)
  subprocess.run(f"source {bbdir}/env.sh", shell=True, check=True)
  subprocess.run(f"cd {arch_dir} && mill -i __.test.runMain Elaborate " 
                 f"--disable-annotation-unknown -strip-debug-info -O=debug "
                 f"--split-verilog -o={build_dir}", shell=True, check=True)
  
  # Remove unwanted file  
  topname_file = f"{arch_dir}/TestHarness.sv"
  if os.path.exists(topname_file):
    os.remove(topname_file)
  
  # For run workflow, continue to build; for standalone verilog, complete
  if data.get("from_run_workflow"):
    await context.emit({"topic": "verilator.build", "data": data})
  else:
    await context.emit({"topic": "verilator.complete", "data": {**data, "task": "verilog"}})