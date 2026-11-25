// The cycles of QUANT_F32_TO_I8 and QUANT_I32_TO_I8: their BEMU cycle model
// (emu/src/latency.rs).
class toint8_cycles extends bb_blink_cycles #(`BB_IN_BW, `BB_OUT_BW);
  `uvm_component_utils(toint8_cycles)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function longint unsigned expected(bb_blink_cmd_item cmd);
    return toint8_bemu_cycles(cmd.iter, cmd.op1_col);
  endfunction
endclass
