import os
import subprocess
import glob

config = {
  "type": "event",
  "name": "build-verilator",
  "description": "compile verilog to verilator executable",
  "subscribes": ["build.verilator"],
  "emits": ["build.simulation", "build.verilator.completed"],
  "input": { 
    "type": "object", 
    "properties": {
      "debug": {"type": "boolean"},
      "workload": {"type": "string"},
      "clean_first": {"type": "boolean"},
      "target": {"type": "string"},
      "trace_id": {"type": "string"}
    }
  },
  "flows": ["build-verilator"],
}

async def handler(input_data, context):
  context.logger.info('Starting verilator build', {"trace_id": input_data.get("trace_id")})
  
  try:
    # 获取项目根目录和相关路径
    bbdir = os.popen('git rev-parse --show-toplevel').read().strip()
    arch_dir = os.path.join(bbdir, 'arch')
    build_dir = os.path.join(arch_dir, 'build')
    obj_dir = os.path.join(build_dir, 'obj_dir')
    
    # 检查verilog文件是否存在
    vsrcs = glob.glob(os.path.join(build_dir, "*.v")) + glob.glob(os.path.join(build_dir, "*.sv"))
    if not vsrcs:
      raise Exception("No verilog files found in build directory")
    
    context.logger.info('Found verilog sources', {"count": len(vsrcs), "files": vsrcs[:5]})  # 只显示前5个
    
    # 获取C/C++源文件
    csrc_paths = [
      os.path.join(arch_dir, 'src', 'csrc'),
      build_dir
    ]
    csrcs = []
    for path in csrc_paths:
      if os.path.exists(path):
        csrcs.extend(glob.glob(os.path.join(path, "*.c")))
        csrcs.extend(glob.glob(os.path.join(path, "*.cc")))
        csrcs.extend(glob.glob(os.path.join(path, "*.cpp")))
    
    context.logger.info('Found C/C++ sources', {"count": len(csrcs)})
    
    # 设置编译参数
    topname = "TestHarness"
    num_jobs = "16"
    debug_mode = input_data.get("debug", False)
    
    # Include路径
    inc_paths = [
      os.path.join(arch_dir, 'src', 'csrc', 'include'),
      '/usr/lib/llvm-11/include',
      os.path.join(bbdir, 'thirdparty', 'chipyard', 'toolchains', 'riscv-tools', 'riscv-isa-sim'),
      os.path.join(bbdir, 'thirdparty', 'chipyard', 'tools', 'DRAMSim2'),
      build_dir
    ]
    
    # 清理之前的obj目录
    if os.path.exists(obj_dir):
      import shutil
      shutil.rmtree(obj_dir)
    
    # 构建verilator命令
    verilator_cmd = [
      "verilator",
      "--build", "-cc", "--trace",
      "-O3", "--x-assign", "fast", "--x-initial", "fast", "--noassert",
      "-Wno-fatal",
      "-j", num_jobs,
      "--timing",
      f"+incdir+{build_dir}",
      "--top", topname
    ]
    
    # 添加verilog源文件
    verilator_cmd.extend(vsrcs)
    verilator_cmd.extend(csrcs)
    
    # 添加编译标志
    cflags = []
    for inc_path in inc_paths:
      if os.path.exists(inc_path):
        cflags.extend(["-I", inc_path])
    
    cflags.extend([
      f'-DTOP_NAME="V{topname}"',
      "-std=c++17"
    ])
    
    if debug_mode:
      cflags.extend(["-Og", "-ggdb3"])
    
    # 添加链接标志
    ldflags = [
      "-lreadline", "-ldramsim", "-lfesvr",
      f"-L{os.path.join(bbdir, 'thirdparty', 'chipyard', 'tools', 'DRAMSim2')}",
      f"-L{os.path.join(bbdir, 'thirdparty', 'chipyard', 'toolchains', 'riscv-tools', 'riscv-isa-sim', 'build')}",
      f"-L{os.path.join(bbdir, 'thirdparty', 'chipyard', 'toolchains', 'riscv-tools', 'riscv-isa-sim', 'build', 'lib')}"
    ]
    
    # 添加CFLAGS和LDFLAGS到verilator命令
    for cflag in cflags:
      verilator_cmd.extend(["-CFLAGS", cflag])
    
    for ldflag in ldflags:
      verilator_cmd.extend(["-LDFLAGS", ldflag])
    
    verilator_cmd.extend(["--Mdir", obj_dir, "--exe"])
    
    context.logger.info('Running verilator command', {"cmd_length": len(verilator_cmd)})
    
    # 执行verilator命令
    result = subprocess.run(
      verilator_cmd,
      cwd=arch_dir,
      capture_output=True,
      text=True,
      timeout=600  # 10分钟超时
    )
    
    if result.returncode != 0:
      context.logger.error('Verilator command failed', {
        "returncode": result.returncode,
        "stdout": result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout,
        "stderr": result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr
      })
      raise Exception(f"Verilator command failed with return code {result.returncode}")
    
    context.logger.info('Verilator generation completed, starting make')
    
    # 执行make命令构建最终可执行文件
    make_cmd = [
      "make", "-C", obj_dir,
      "-f", f"V{topname}.mk",
      f"{obj_dir}/V{topname}"
    ]
    
    result = subprocess.run(
      make_cmd,
      cwd=arch_dir,
      capture_output=True,
      text=True,
      timeout=300  # 5分钟超时
    )
    
    if result.returncode != 0:
      context.logger.error('Make command failed', {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr
      })
      raise Exception(f"Make command failed with return code {result.returncode}")
    
    # 检查可执行文件是否生成
    executable = os.path.join(obj_dir, f"V{topname}")
    if not os.path.exists(executable):
      raise Exception(f"Executable {executable} was not created")
    
    context.logger.info('Verilator build completed successfully', {
      "executable": executable,
      "size": os.path.getsize(executable)
    })
    
    # 发出verilator编译完成事件
    await context.emit({
      "topic": "build.verilator.completed", 
      "data": {
        **input_data,
        "executable": executable
      }
    })
    
    # 根据target决定下一步
    target = input_data.get("target", "run")
    if target in ["run", "sim"]:
      # 继续到仿真步骤
      await context.emit({
        "topic": "build.simulation", 
        "data": {
          **input_data,
          "executable": executable
        }
      })
    
  except subprocess.TimeoutExpired:
    context.logger.error('Verilator build timed out', {
      "trace_id": input_data.get("trace_id")
    })
    await context.emit({
      "topic": "build.error", 
      "data": {
        **input_data,
        "error": "Verilator build timed out",
        "step": "verilator"
      }
    })
    raise
  except Exception as e:
    context.logger.error('Verilator build failed', {
      "error": str(e),
      "trace_id": input_data.get("trace_id")
    })
    await context.emit({
      "topic": "build.error", 
      "data": {
        **input_data,
        "error": str(e),
        "step": "verilator"
      }
    })
    raise