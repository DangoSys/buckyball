#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
from pathlib import Path
import glob

def run_cmd(cmd, cwd=None):
  print(f"Running: {cmd}")
  subprocess.run(cmd, shell=True, check=True, cwd=cwd)

def get_paths():
  bbdir = subprocess.check_output("git rev-parse --show-toplevel", shell=True).decode().strip()
  archdir = f"{bbdir}/arch"
  build_dir = f"{archdir}/build"
  return bbdir, archdir, build_dir

def setup_dirs(archdir, build_dir):
  os.makedirs(build_dir, exist_ok=True)
  os.makedirs(f"{archdir}/waveform", exist_ok=True)  
  os.makedirs(f"{archdir}/log", exist_ok=True)

def find_sources(archdir, build_dir):
  vsrcs = glob.glob(f"{build_dir}/**/*.v", recursive=True) + glob.glob(f"{build_dir}/**/*.sv", recursive=True)
  csrcs = (glob.glob(f"{archdir}/src/csrc/**/*.c", recursive=True) + 
           glob.glob(f"{archdir}/src/csrc/**/*.cc", recursive=True) +
           glob.glob(f"{archdir}/src/csrc/**/*.cpp", recursive=True) +
           glob.glob(f"{build_dir}/**/*.c", recursive=True) +
           glob.glob(f"{build_dir}/**/*.cc", recursive=True) +
           glob.glob(f"{build_dir}/**/*.cpp", recursive=True))
  return vsrcs, csrcs

def build_args(binary):
  if binary:
    return f"+permissive +loadmem={binary} +loadmem_addr=80000000 +custom_boot_pin=1 +permissive-off {binary} "
  return ""

def verilog(archdir, build_dir):
  setup_dirs(archdir, build_dir)
  run_cmd(f"mill -i __.test.runMain Elaborate --disable-annotation-unknown -strip-debug-info -O=debug --split-verilog -o={build_dir}")
  topname_file = f"{archdir}/TestHarness.sv"
  if os.path.exists(topname_file):
    os.remove(topname_file)

def build_verilator(archdir, build_dir, num_jobs=16):
  inc_paths = [
    os.environ.get('RISCV', '') + '/include' if os.environ.get('RISCV') else '',
    f"{archdir}/thirdparty/chipyard/tools/DRAMSim2",
    build_dir,
    f"{archdir}/src/csrc/include"
  ]
  inc_flags = ' '.join([f"-I{p}" for p in inc_paths if p])
  
  cflags = f"{inc_flags} -DTOP_NAME='\"VTestHarness\"' -std=c++17"
  ldflags = (f"-lreadline -ldramsim -lfesvr "
             f"-L{archdir}/thirdparty/chipyard/tools/DRAMSim2 "
             f"-L{archdir}/thirdparty/chipyard/toolchains/riscv-tools/riscv-isa-sim/build "
             f"-L{archdir}/thirdparty/chipyard/toolchains/riscv-tools/riscv-isa-sim/build/lib")
  
  obj_dir = f"{build_dir}/obj_dir"
  run_cmd(f"rm -rf {obj_dir}")
  
  vsrcs, csrcs = find_sources(archdir, build_dir)
  sources = ' '.join(vsrcs + csrcs)

  verilator_cmd = (f"verilator -MMD --build -cc --trace -O3 --x-assign fast --x-initial fast --noassert -Wno-fatal "
                   f"--timing -j {num_jobs} +incdir+{build_dir} --top TestHarness {sources} "
                   f"-CFLAGS '{cflags}' -LDFLAGS '{ldflags}' --Mdir {obj_dir} --exe")
  
  run_cmd(verilator_cmd)
  run_cmd(f"make -C {obj_dir} -f VTestHarness.mk {obj_dir}/VTestHarness")

def sim(archdir, build_dir, args):
  bin_path = f"{build_dir}/obj_dir/VTestHarness"
  run_cmd(f"{bin_path} {args}")

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('command', choices=['run', 'test', 'verilog', 'compile', 'clean', 'sim', 'wave', 'help', 'bsp', 'reformat', 'checkformat'])
  parser.add_argument('--binary', help='Binary file path for simulation')
  parser.add_argument('--jobs', type=int, default=16, help='Number of parallel jobs')
  args = parser.parse_args()
  
  bbdir, archdir, build_dir = get_paths()
  
  if args.command == 'run':
    verilog(archdir, build_dir)
    build_verilator(archdir, build_dir, args.jobs)
    sim_args = build_args(args.binary)
    sim(archdir, build_dir, sim_args)
    
  elif args.command == 'verilog':
    verilog(archdir, build_dir)
    
  elif args.command == 'sim':
    sim_args = build_args(args.binary)
    sim(archdir, build_dir, sim_args)
    
  elif args.command == 'test':
    run_cmd("mill -i __.test")
    
  elif args.command == 'compile':
    run_cmd("mill -i __.compile")
    
  elif args.command == 'clean':
    run_cmd(f"rm -rf {build_dir}")
    
  elif args.command == 'wave':
    run_cmd("gtkwave dump.vcd &")
    
  elif args.command == 'help':
    run_cmd("mill -i __.test.runMain Elaborate --help")
    
  elif args.command == 'bsp':
    run_cmd("mill -i mill.bsp.BSP/install")
    
  elif args.command == 'reformat':
    run_cmd("mill -i __.reformat")
    
  elif args.command == 'checkformat':
    run_cmd("mill -i __.checkFormat")

if __name__ == "__main__":
  main()