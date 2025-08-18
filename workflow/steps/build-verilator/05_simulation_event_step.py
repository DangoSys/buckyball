import os
import subprocess

config = {
  "type": "event",
  "name": "build-simulation",
  "description": "run verilator simulation with configurable workload",
  "subscribes": ["build.simulation"],
  "emits": ["build-verilator-finished", "build.simulation.completed"],
  "input": { 
    "type": "object", 
    "properties": {
      "debug": {"type": "boolean"},
      "workload": {"type": "string"},
      "clean_first": {"type": "boolean"},
      "target": {"type": "string"},
      "trace_id": {"type": "string"},
      "executable": {"type": "string"}
    }
  },
  "flows": ["build-verilator"],
}

async def handler(input_data, context):
  context.logger.info('Starting simulation', {"trace_id": input_data.get("trace_id")})
  
  try:
    # 获取项目根目录和相关路径
    bbdir = os.popen('git rev-parse --show-toplevel').read().strip()
    arch_dir = os.path.join(bbdir, 'arch')
    log_dir = os.path.join(arch_dir, 'log')
    
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 获取可执行文件路径
    executable = input_data.get("executable")
    if not executable or not os.path.exists(executable):
      raise Exception(f"Executable not found: {executable}")
    
    context.logger.info('Found executable', {"executable": executable})
    
    # 构建仿真参数
    workload = input_data.get("workload", "")
    debug_mode = input_data.get("debug", False)
    
    # 基础参数
    args = ["+permissive"]
    
    # 如果指定了workload，使用指定的workload
    if workload:
      if not workload.startswith("/"):
        # 相对路径，添加默认前缀
        default_workload_base = "/home/mio/Code/buckyball/bb-tests/workloads/build/src/CTest"
        workload_path = os.path.join(default_workload_base, workload)
      else:
        workload_path = workload
      
      # 检查workload文件是否存在
      if os.path.exists(workload_path):
        args.extend([
          f"+loadmem={workload_path}",
          "+loadmem_addr=80000000",
          "+custom_boot_pin=1",
          "+permissive-off",
          workload_path,
          "-b"
        ])
        context.logger.info('Using specified workload', {"workload": workload_path})
      else:
        context.logger.warning('Workload file not found, using default', {"workload": workload_path})
        # 使用默认workload
        default_workload = "/home/mio/Code/buckyball/bb-tests/workloads/build/src/CTest/ctest_vecunit_matmul_16xn_zero_random_singlecore-baremetal"
        args.extend([
          f"+loadmem={default_workload}",
          "+loadmem_addr=80000000",
          "+custom_boot_pin=1",
          "+permissive-off",
          default_workload,
          "-b"
        ])
    else:
      # 使用默认workload
      default_workload = "/home/mio/Code/buckyball/bb-tests/workloads/build/src/CTest/ctest_vecunit_matmul_16xn_zero_random_singlecore-baremetal"
      args.extend([
        f"+loadmem={default_workload}",
        "+loadmem_addr=80000000",
        "+custom_boot_pin=1",
        "+permissive-off",
        default_workload,
        "-b"
      ])
    
    # 调试模式相关参数
    if debug_mode:
      args.extend(["+uart_tx=1", "+uart_tx_printf=1"])
    
    # 构建完整的仿真命令
    sim_cmd = [executable] + args
    
    context.logger.info('Running simulation', {
      "cmd": " ".join(sim_cmd),
      "workdir": arch_dir
    })
    
    # 设置环境变量
    env = os.environ.copy()
    
    # 运行仿真
    result = subprocess.run(
      sim_cmd,
      cwd=arch_dir,
      env=env,
      capture_output=True,
      text=True,
      timeout=1800  # 30分钟超时
    )
    
    # 记录仿真结果
    context.logger.info('Simulation completed', {
      "returncode": result.returncode,
      "stdout_length": len(result.stdout),
      "stderr_length": len(result.stderr)
    })
    
    # 保存仿真输出到日志文件
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"simulation_{timestamp}.log")
    
    with open(log_file, 'w') as f:
      f.write(f"=== Simulation Log - {timestamp} ===\n")
      f.write(f"Command: {' '.join(sim_cmd)}\n")
      f.write(f"Return code: {result.returncode}\n")
      f.write(f"Workload: {workload or 'default'}\n")
      f.write(f"Debug mode: {debug_mode}\n")
      f.write("\n=== STDOUT ===\n")
      f.write(result.stdout)
      f.write("\n=== STDERR ===\n")
      f.write(result.stderr)
    
    context.logger.info('Simulation log saved', {"log_file": log_file})
    
    # 检查仿真是否成功（根据返回码判断）
    if result.returncode == 0:
      context.logger.info('Simulation completed successfully')
      
      # 发出仿真完成事件
      await context.emit({
        "topic": "build.simulation.completed", 
        "data": {
          **input_data,
          "log_file": log_file,
          "returncode": result.returncode,
          "success": True
        }
      })
    else:
      context.logger.warning('Simulation completed with non-zero exit code', {
        "returncode": result.returncode,
        "stderr_preview": result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
      })
      
      # 即使返回码非零，也可能是正常的仿真结束，发出完成事件
      await context.emit({
        "topic": "build.simulation.completed", 
        "data": {
          **input_data,
          "log_file": log_file,
          "returncode": result.returncode,
          "success": False,
          "stderr": result.stderr
        }
      })
    
    # 发出workflow结束事件
    await context.emit({
      "topic": "build-verilator-finished", 
      "data": {
        **input_data,
        "final_status": "completed",
        "log_file": log_file
      }
    })
    
  except subprocess.TimeoutExpired:
    context.logger.error('Simulation timed out', {
      "trace_id": input_data.get("trace_id")
    })
    await context.emit({
      "topic": "build.error", 
      "data": {
        **input_data,
        "error": "Simulation timed out",
        "step": "simulation"
      }
    })
    # 即使超时也发出结束事件
    await context.emit({
      "topic": "build-verilator-finished", 
      "data": {
        **input_data,
        "final_status": "timeout"
      }
    })
    raise
  except Exception as e:
    context.logger.error('Simulation failed', {
      "error": str(e),
      "trace_id": input_data.get("trace_id")
    })
    await context.emit({
      "topic": "build.error", 
      "data": {
        **input_data,
        "error": str(e),
        "step": "simulation"
      }
    })
    # 即使失败也发出结束事件
    await context.emit({
      "topic": "build-verilator-finished", 
      "data": {
        **input_data,
        "final_status": "failed",
        "error": str(e)
      }
    })
    raise