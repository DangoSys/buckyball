// TRANSPOSE's cycles: its BEMU cycle model (emu/src/latency.rs).
class transpose_cycles extends bb_blink_cycles #(`BB_IN_BW, `BB_OUT_BW);
  `uvm_component_utils(transpose_cycles)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function longint unsigned expected(bb_blink_cmd_item cmd);
    return transpose_bemu_cycles(cmd.iter, cmd.rs2, cmd.op1_col);
  endfunction
endclass
