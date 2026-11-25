// VECMAT16's cycles: its BEMU cycle model (emu/src/latency.rs).
class vector_cycles extends bb_blink_cycles #(`BB_IN_BW, `BB_OUT_BW);
  `uvm_component_utils(vector_cycles)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function longint unsigned expected(bb_blink_cmd_item cmd);
    return vector_bemu_cycles(cmd.iter);
  endfunction
endclass
