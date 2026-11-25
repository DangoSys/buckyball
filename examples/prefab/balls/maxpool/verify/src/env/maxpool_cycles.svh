// MAXPOOL and TOPK_U32's cycles: its BEMU cycle model (emu/src/latency.rs).
class maxpool_cycles extends bb_blink_cycles #(`BB_IN_BW, `BB_OUT_BW);
  `uvm_component_utils(maxpool_cycles)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function longint unsigned expected(bb_blink_cmd_item cmd);
    if (cmd.funct == TOPK_THREAD_FUNCT7[6:0])
      return topk_bemu_cycles(cmd.iter, cmd.op1_col, cmd.rs2);
    return maxpool_bemu_cycles(cmd.rs2, cmd.op1_col);
  endfunction
endclass
