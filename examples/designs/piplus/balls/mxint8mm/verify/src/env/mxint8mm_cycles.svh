// MXINT8MM's cycles: its BEMU cycle model (emu/src/latency.rs) on this array.
class mxint8mm_cycles extends bb_blink_cycles #(`BB_IN_BW, `BB_OUT_BW);
  `uvm_component_utils(mxint8mm_cycles)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function longint unsigned expected(bb_blink_cmd_item cmd);
    return mxint8mm_bemu_cycles(cmd.iter, cmd.rs2, `MXINT8MM_ROWS, `MXINT8MM_COLS, `MXINT8MM_LANES);
  endfunction
endclass
