import os
import subprocess

config = {
  "type": "event",
  "name": "build-verilog",
  "description": "generate verilog code using mill",
  "subscribes": ["build.verilog"],
  "emits": ["build.verilator", "build.verilog.completed"],
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
  context.logger.info('Starting verilog generation', {"trace_id": input_data.get("trace_id")})
  
  try:
    # 获取项目根目录和构建目录
    bbdir = os.popen('git rev-parse --show-toplevel').read().strip()
    build_dir = os.path.join(bbdir, 'arch', 'build')
    arch_dir = os.path.join(bbdir, 'arch')
    
    # 确保构建目录存在
    os.makedirs(build_dir, exist_ok=True)
    
    context.logger.info('Generating verilog code', {"build_dir": build_dir})
    
    # 构建mill命令
    debug_mode = input_data.get("debug", False)
    optimization = "-O=debug" if debug_mode else "-O=release"
    
    mill_cmd = [
      "mill", "-i", "__.test.runMain", "Elaborate",
      "--disable-annotation-unknown",
      "-strip-debug-info",
      optimization,
      "--split-verilog",
      f"-o={build_dir}"
    ]
    
    # 设置环境变量
    env = os.environ.copy()
    env["PATH"] = f"{env['PATH']}:{os.path.join(bbdir, 'tools', 'mill')}"
    
    context.logger.info('Running mill command', {"cmd": " ".join(mill_cmd)})
    
    # 执行mill命令
    result = subprocess.run(
      mill_cmd,
      cwd=arch_dir,
      env=env,
      capture_output=True,
      text=True,
      timeout=300  # 5分钟超时
    )
    
    if result.returncode != 0:
      context.logger.error('Mill command failed', {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr
      })
      raise Exception(f"Mill command failed with return code {result.returncode}: {result.stderr}")
    
    context.logger.info('Verilog generation completed successfully', {
      "stdout": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout  # 只记录最后500字符
    })
    
    # 清理可能生成的多余文件
    topname_sv = os.path.join(arch_dir, "TestHarness.sv")
    if os.path.exists(topname_sv):
      os.remove(topname_sv)
      context.logger.info('Removed redundant TestHarness.sv file')
    
    # 发出verilog生成完成事件
    await context.emit({
      "topic": "build.verilog.completed", 
      "data": input_data
    })
    
    # 根据target决定下一步
    target = input_data.get("target", "run")
    if target in ["run", "sim"]:
      # 继续到verilator编译步骤
      await context.emit({
        "topic": "build.verilator", 
        "data": input_data
      })
    
  except subprocess.TimeoutExpired:
    context.logger.error('Verilog generation timed out', {
      "trace_id": input_data.get("trace_id")
    })
    await context.emit({
      "topic": "build.error", 
      "data": {
        **input_data,
        "error": "Verilog generation timed out",
        "step": "verilog"
      }
    })
    raise
  except Exception as e:
    context.logger.error('Verilog generation failed', {
      "error": str(e),
      "trace_id": input_data.get("trace_id")
    })
    await context.emit({
      "topic": "build.error", 
      "data": {
        **input_data,
        "error": str(e),
        "step": "verilog"
      }
    })
    raise