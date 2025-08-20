config = {
  "type": "noop",
  "name": "build verilator",
  "description": "compile chisel source code to verilator for simulation",
  "virtualSubscribes": [],
  "virtualEmits": ["/verilator"],
  "flows": ["verilator"],
}