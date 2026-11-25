// MXMM's cycles: its cycle model (casegen/src/latency.rs).
class mxmm_cycles extends bb_blink_cycles #(`BB_IN_BW, `BB_OUT_BW);
  `uvm_component_utils(mxmm_cycles)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function longint unsigned expected(bb_blink_cmd_item cmd);
    return mxmm_bemu_cycles(cmd.funct == `MXMM_MXFP8_WINDOW_FUNCT7, cmd.iter, cmd.rs2);
  endfunction
endclass
