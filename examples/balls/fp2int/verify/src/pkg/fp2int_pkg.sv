package fp2int_pkg;
  import uvm_pkg::*;
  import bb_uvm_pkg::*;
  `include "uvm_macros.svh"

  `include "common/fp2int_defs.svh"
  `include "common/fp2int_items.svh"
  `include "seq/fp2int_sequences.svh"
  `include "agents/cmd/fp2int_cmd_driver.svh"
  `include "agents/cmd/fp2int_cmd_monitor.svh"
  `include "agents/cmd/fp2int_cmd_agent.svh"
  `include "agents/mem/fp2int_mem_model.svh"
  `include "agents/mem/fp2int_mem_monitor.svh"
  `include "agents/resp/fp2int_resp_monitor.svh"
  `include "env/fp2int_scoreboard.svh"
  `include "env/fp2int_env.svh"
  `include "tests/fp2int_test.svh"
endpackage
