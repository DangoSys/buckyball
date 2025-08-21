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
  "name": "build-verilog", 
  "description": "generate verilog code",
  "subscribes": ["build.verilog"],
  "emits": ["build.verilator", "build.error"],
  "flows": ["verilator"],
}

async def handler(data, context):
  bbdir = get_buckyball_path()
  build_dir = f"{bbdir}/arch/build"
  arch_dir = f"{bbdir}/arch"
  
  os.makedirs(build_dir, exist_ok=True)
  subprocess.run(f"source {bbdir}/env.sh", shell=True, check=True)
  subprocess.run(f"cd {arch_dir} && mill -i __.test.runMain Elaborate " 
                 f"--disable-annotation-unknown -strip-debug-info -O=debug --split-verilog -o={build_dir}", 
                 shell=True, check=True)
  
  # Remove unwanted file  
  topname_file = f"{arch_dir}/TestHarness.sv"
  if os.path.exists(topname_file):
    os.remove(topname_file)
  
  if data.get("target") in ["run", "sim"]:
    await context.emit({"topic": "build.verilator", "data": data})