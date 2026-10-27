class im2col_cov extends uvm_component;
  `uvm_component_utils(im2col_cov)

  uvm_analysis_imp_cmd #(bb_blink_cmd_item, im2col_cov) cmd_imp;

  covergroup cmd_cg;
    coverpoint kernel {bins k1 = {1}; bins k2 = {2}; bins k3 = {3}; bins k5 = {5}; bins k7 = {7};}
    coverpoint stride {bins s1 = {1}; bins s2 = {2};}
    coverpoint padding {bins p0 = {0}; bins p1 = {1};}
    coverpoint start {bins zero = {0}; bins shifted = {[1 : $]};}
    coverpoint input_base {bins zero = {0}; bins offset = {[1 : $]};}
    coverpoint window_start {bins zero = {0}; bins offset = {[1 : $]};}
    coverpoint tile_rows {
      bins t1 = {1}; bins t2 = {2}; bins t4 = {4}; bins t8 = {8}; bins t16 = {16};
    }
    coverpoint in_groups {bins one = {1}; bins two = {2};}
    coverpoint out_groups {bins one = {1}; bins many = {[2 : $]};}
  endgroup

  int unsigned kernel;
  int unsigned stride;
  int unsigned padding;
  int unsigned start;
  int unsigned input_base;
  int unsigned window_start;
  int unsigned tile_rows;
  int unsigned in_groups;
  int unsigned out_groups;
  int cmd_count;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    cmd_imp = new("cmd_imp", this);
    cmd_cg  = new();
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    kernel       = int'(item.rs2[3:0]);
    stride       = int'(item.rs2[7:4]);
    padding      = int'(item.rs2[11:8]);
    start        = int'(item.rs2[19:12]);
    input_base   = int'(item.rs2[25:20]);
    window_start = int'(item.rs2[35:26]);
    tile_rows    = int'(item.rs2[50:46]);
    in_groups    = int'(item.op1_col);
    out_groups   = int'(item.wr_col);
    cmd_count++;
    cmd_cg.sample();
  endfunction

  function void check_phase(uvm_phase phase);
    verification_pkg::evidence::checks++;
    super.check_phase(phase);
    if (cmd_count == 0) `uvm_fatal("COV", "no cmd observed")
    if (cmd_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf("im2col_cov coverage incomplete: %0.2f%%", cmd_cg.get_coverage()))
  endfunction
endclass
