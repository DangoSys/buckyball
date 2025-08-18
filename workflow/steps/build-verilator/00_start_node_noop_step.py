config = {
  "type": "noop",
  "name": "build verilator",
  "description": "build chisel to verilator for simulation",
  "virtualSubscribes": [],
  "virtualEmits": ["/build-verilator"],
  "flows": ["build-verilator"],
}