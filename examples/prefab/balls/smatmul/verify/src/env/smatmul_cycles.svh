// SMatMul's cycles: its BEMU cycle model (emu/src/latency.rs).
class smatmul_cycles extends bb_blink_cycles #(`BB_IN_BW, `BB_OUT_BW);
  `uvm_component_utils(smatmul_cycles)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function longint unsigned expected(bb_blink_cmd_item cmd);
    longint unsigned tile_rows = `SMATMUL_TILE_ROWS;
    longint unsigned tile_cols = `SMATMUL_TILE_COLS;
    return smatmul_bemu_cycles(
        cmd.funct == SMATMUL_BIAS_THREAD_FUNCT7[6:0],
        cmd.iter,
        cmd.rs2,
        cmd.op1_col,
        tile_rows,
        tile_cols
    );
  endfunction
endclass
