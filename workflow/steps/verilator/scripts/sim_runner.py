import os
import subprocess
import sys
from datetime import datetime

# Add the utils directory to the Python path
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if utils_path not in sys.path:
  sys.path.insert(0, utils_path)

from utils.path import get_buckyball_path

def run_simulation(binary, batch=False, logger=None):
  """运行verilator仿真的普通函数"""
  if logger:
    logger.info('开始执行仿真', {"binary": binary, "batch": batch})
  
  bbdir = get_buckyball_path()
  arch_dir = f"{bbdir}/arch"
  build_dir = f"{arch_dir}/build"
  timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
  
  binary_name = os.path.basename(binary) if binary else "no_binary"
  log_dir = f"{arch_dir}/log/{timestamp}-{binary_name}"
  waveform_dir = f"{arch_dir}/waveform"
  waveform_fst_dir = f"{arch_dir}/waveform/{timestamp}-{binary_name}"
  
  os.makedirs(log_dir, exist_ok=True)
  os.makedirs(waveform_dir, exist_ok=True)
  
  bin_path = f"{build_dir}/obj_dir/VTestHarness"
  
  # 清理旧波形文件
  subprocess.run(f"rm -f {waveform_dir}/waveform.vcd {waveform_dir}/waveform.fst", shell=True, check=True)
  
  # 执行仿真
  command = f"source {bbdir}/env.sh && ./sim.sh {bin_path} {binary} {log_dir}/stdout.log {log_dir}/disasm.log {batch}"
  if logger:
    logger.info('执行仿真命令', {'command': command})
  
  result = subprocess.run(command, cwd=os.path.dirname(__file__), shell=True, text=True)
  if logger:
    logger.info('仿真执行完成', {"return_code": result.returncode})

  # 转换波形
  os.makedirs(waveform_fst_dir, exist_ok=True)
  subprocess.run(f"vcd2fst -v {waveform_dir}/waveform.vcd -f {waveform_fst_dir}/waveform.fst", cwd=arch_dir, shell=True, check=True, text=True)
  if logger:
    logger.info('波形转换完成')
  
  return {
    "success": result.returncode == 0,
    "return_code": result.returncode,
    "log_dir": log_dir,
    "waveform_dir": waveform_fst_dir,
    "completed_at": datetime.now().isoformat()
  }