// MXINT8QUANT's cycles: its BEMU cycle model (emu/src/latency.rs).
class mxint8quant_cycles extends bb_blink_cycles #(`BB_IN_BW, `BB_OUT_BW);
  `uvm_component_utils(mxint8quant_cycles)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function longint unsigned expected(bb_blink_cmd_item cmd);
    return mxint8quant_bemu_cycles(cmd.iter, cmd.rs2);
  endfunction
endclass
