#!/usr/bin/env python3

import subprocess
from pathlib import Path


def main():
  test_dir = Path(__file__).resolve().parent
  tests = [
    "test_api_compiler_build.py",
    "test_api_workload_build.py",
    "test_api_verilator_clean.py",
    "test_api_verilator_verilog.py",
    "test_api_verilator_build.py",
    "test_api_verilator_sim.py",
    "test_api_verilator_run.py",
    "test_api_yosys_run.py",
    "test_api_yosys_verilog.py",
    "test_api_yosys_synth.py",
    "test_api_dc_verilog.py",
    "test_api_kernel_build.py",
    "test_api_firesim_enumeratefpgas.py",
    "test_api_firesim_buildbitstream.py",
    "test_api_firesim_infrasetup.py",
    "test_api_firesim_runworkload.py",
    "test_api_bebop_bemu_build.py",
    "test_api_bebop_bemu_sim.py",
    "test_api_bebop_bemu_batch.py",
    "test_api_bebop_verilator_clean.py",
    "test_api_bebop_verilator_verilog.py",
    "test_api_bebop_verilator_build.py",
    "test_api_bebop_verilator_sim.py",
    "test_api_bebop_verilator_run.py",
    "test_api_bebop_verilator_batch.py",
    "test_api_bebop_p2e_clean.py",
    "test_api_bebop_p2e_verilog.py",
    "test_api_bebop_p2e_buildbitstream.py",
    "test_api_bebop_p2e_runworkload.py",
    "test_api_bebop_p2e_batch.py",
  ]
  for name in tests:
    path = test_dir / name
    if not path.exists():
      raise RuntimeError(f"missing test: {name}")
    print(f"RUN {name}")
    ret = subprocess.run(["python", str(path)], cwd=str(test_dir))
    if ret.returncode != 0:
      raise RuntimeError(f"failed: {name}")
  print(f"PASS: {len(tests)} api tests done.")


if __name__ == "__main__":
  main()
