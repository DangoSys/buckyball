// F32ADD cycles: the BEMU cycle model (emu/src/latency.rs).
class f32add_cycles extends bb_blink_cycles #(`BB_IN_BW, `BB_OUT_BW);
  `uvm_component_utils(f32add_cycles)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function longint unsigned expected(bb_blink_cmd_item cmd);
    return f32add_bemu_cycles(cmd.iter);
  endfunction
endclass
