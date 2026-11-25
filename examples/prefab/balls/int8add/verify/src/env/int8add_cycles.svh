// INT8ADD and INT8ADD_RELU cycles: the BEMU cycle model (emu/src/latency.rs).
class int8add_cycles extends bb_blink_cycles #(`BB_IN_BW, `BB_OUT_BW);
  `uvm_component_utils(int8add_cycles)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function longint unsigned expected(bb_blink_cmd_item cmd);
    return int8add_bemu_cycles(cmd.iter, cmd.op1_col);
  endfunction
endclass
