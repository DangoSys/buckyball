import os
from datetime import datetime


def req_arg(body: dict, name: str):
  return body.get(name) or body.get(name.replace("_", "-"))


def make_yosys_log_dir(bbdir: str, trace_id: str) -> str:
  stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
  suffix = trace_id[:8] if trace_id else "no-trace"
  return os.path.join(bbdir, "bbdev", "api", "steps", "yosys", "log", f"{stamp}-{suffix}")
