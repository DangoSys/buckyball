config = {
  "type": "api",
  "name": "build-verilator-api",
  "description": "trigger build-verilator flow with configurable parameters",
  "path": "/build-verilator",
  "method": "POST",
  "emits": ["build.clean", "build.verilog", "build.run"],
  "virtualSubscribes": ["/build-verilator"],
  "flows": ["build-verilator"],
}

async def handler(req, context):
  body = req.get("body") or {}
  context.logger.info('build-verilator API received', { "body": body })
  
  # 提取参数
  debug_mode = body.get("debug", False)
  workload = body.get("workload", "")
  clean_first = body.get("clean", True)
  target = body.get("target", "run")  # run, verilog, sim, test等
  
  # 构建配置
  build_config = {
    "debug": debug_mode,
    "workload": workload,
    "clean_first": clean_first,
    "target": target,
    "trace_id": context.trace_id
  }
  
  # 根据目标发出不同的事件
  if target == "clean":
    await context.emit({"topic": "build.clean", "data": build_config})
  elif target == "verilog":
    if clean_first:
      await context.emit({"topic": "build.clean", "data": build_config})
    else:
      await context.emit({"topic": "build.verilog", "data": build_config})
  elif target in ["run", "sim"]:
    if clean_first:
      await context.emit({"topic": "build.clean", "data": build_config})
    else:
      await context.emit({"topic": "build.verilog", "data": build_config})
  else:
    # 默认运行完整流程
    if clean_first:
      await context.emit({"topic": "build.clean", "data": build_config})
    else:
      await context.emit({"topic": "build.verilog", "data": build_config})
  
  return {"status": 200, "body": {
    "traceId": context.trace_id, 
    "message": f"build-verilator {target} emitted",
    "config": build_config
  }}

