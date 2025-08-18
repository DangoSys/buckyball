import os
import shutil

config = {
  "type": "event",
  "name": "build-clean",
  "description": "clean build directory before starting new build",
  "subscribes": ["build.clean"],
  "emits": ["build.verilog", "build.clean.completed"],
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
  context.logger.info('Starting clean process', {"trace_id": input_data.get("trace_id")})
  
  try:
    # 获取项目根目录和构建目录
    bbdir = os.popen('git rev-parse --show-toplevel').read().strip()
    build_dir = os.path.join(bbdir, 'arch', 'build')
    
    context.logger.info('Cleaning build directory', {"build_dir": build_dir})
    
    # 清理构建目录
    if os.path.exists(build_dir):
      shutil.rmtree(build_dir)
      context.logger.info('Build directory cleaned successfully')
    else:
      context.logger.info('Build directory does not exist, skipping clean')
    
    # 发出clean完成事件
    await context.emit({
      "topic": "build.clean.completed", 
      "data": input_data
    })
    
    # 根据target决定下一步
    target = input_data.get("target", "run")
    if target != "clean":
      # 继续到verilog生成步骤
      await context.emit({
        "topic": "build.verilog", 
        "data": input_data
      })
    
  except Exception as e:
    context.logger.error('Clean process failed', {
      "error": str(e),
      "trace_id": input_data.get("trace_id")
    })
    await context.emit({
      "topic": "build.error", 
      "data": {
        **input_data,
        "error": str(e),
        "step": "clean"
      }
    })
    raise