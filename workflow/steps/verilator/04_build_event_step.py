import os
import subprocess
import glob
import sys

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if utils_path not in sys.path:
  sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path

config = {
  "type": "event",
  "name": "make build",
  "description": "build verilator executable", 
  "subscribes": ["verilator.build"],
  "emits": ["verilator.sim", "verilator.complete", "verilator.error"],
  "flows": ["verilator"],
}

async def handler(data, context):
  bbdir = get_buckyball_path()
  arch_dir = f"{bbdir}/arch"
  build_dir = f"{arch_dir}/build"
  waveform_dir = f"{arch_dir}/waveform"
  log_dir = f"{arch_dir}/log"
  
  # Find sources
  vsrcs = glob.glob(f"{build_dir}/**/*.v", recursive=True) + glob.glob(f"{build_dir}/**/*.sv", recursive=True)
  csrcs = (glob.glob(f"{arch_dir}/src/csrc/**/*.c", recursive=True) + 
           glob.glob(f"{arch_dir}/src/csrc/**/*.cc", recursive=True) +
           glob.glob(f"{arch_dir}/src/csrc/**/*.cpp", recursive=True) +
           glob.glob(f"{build_dir}/**/*.c", recursive=True) +
           glob.glob(f"{build_dir}/**/*.cc", recursive=True) +
           glob.glob(f"{build_dir}/**/*.cpp", recursive=True))
  
  # Setup paths
  inc_paths = [
    os.environ.get('RISCV', '') + '/include' if os.environ.get('RISCV') else '',
    f"{arch_dir}/thirdparty/chipyard/tools/DRAMSim2",
    build_dir,
    f"{arch_dir}/src/csrc/include"
  ]
  inc_flags = ' '.join([f"-I{p}" for p in inc_paths if p])
  
  topname = "TestHarness"
  vcd_path = f"{waveform_dir}/waveform.vcd"
  log_path = f"{log_dir}/bdb.log"

  cflags = f"{inc_flags} -DTOP_NAME='\"V{topname}\"' -std=c++17 -DCONFIG_VCD_PATH=\"{vcd_path}\" -DCONFIG_LOG_PATH=\"{log_path}\" "
  ldflags = (f"-lreadline -ldramsim -lfesvr "
             f"-L{arch_dir}/thirdparty/chipyard/tools/DRAMSim2 "
             f"-L{arch_dir}/thirdparty/chipyard/toolchains/riscv-tools/riscv-isa-sim/build "
             f"-L{arch_dir}/thirdparty/chipyard/toolchains/riscv-tools/riscv-isa-sim/build/lib")
  
  obj_dir = f"{build_dir}/obj_dir"
  subprocess.run(f"rm -rf {obj_dir}", shell=True)
  
  sources = ' '.join(vsrcs + csrcs)
  jobs = data.get("jobs", "")
  
  verilator_cmd = (f"verilator -MMD --build -cc --trace -O3 --x-assign fast --x-initial fast --noassert -Wno-fatal "
                   f"--timing -j {jobs} +incdir+{build_dir} --top {topname} {sources} "
                   f"-CFLAGS '{cflags}' -LDFLAGS '{ldflags}' --Mdir {obj_dir} --exe")
  
  subprocess.run(verilator_cmd, shell=True, check=True)
  subprocess.run(f"make -C {obj_dir} -f V{topname}.mk {obj_dir}/V{topname}", shell=True, check=True)
  
  # For run workflow, continue to sim; for standalone build, complete
  if data.get("from_run_workflow"):
    await context.emit({"topic": "verilator.sim", "data": data})
  else:
    await context.emit({"topic": "verilator.complete", "data": {**data, "task": "build"}})
    