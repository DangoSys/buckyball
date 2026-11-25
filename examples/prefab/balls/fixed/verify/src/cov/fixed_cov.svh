class fixed_cov extends uvm_component;
  `uvm_component_utils(fixed_cov)
  uvm_analysis_imp_cmd #(bb_blink_cmd_item, fixed_cov) cmd_imp;
  int mode, fraction, rows, broadcast, stream;
  covergroup cmd_cg;
    m: coverpoint mode {bins modes[] = {[0 : 7]};}
    f: coverpoint fraction {bins fractions[] = {1, 8, 24};}
    r: coverpoint rows {bins sizes[] = {4, 8};}
    b: coverpoint broadcast {bins off = {0}; bins on = {1};}
    s: coverpoint stream {bins off = {0}; bins on = {1};}
    cross m, s{ignore_bins no_stream = binsof (m) intersect {[3 : 7]} && binsof (s) intersect {1};}
    cross m, f, r{ignore_bins mean_rows = binsof (m) intersect {7} && binsof (r) intersect {8};}
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    cmd_imp = new("cmd_imp", this);
    cmd_cg  = new();
  endfunction
  function void write_cmd(bb_blink_cmd_item item);
    mode = item.rs2[2:0];
    fraction = item.rs2[8:4];
    rows = item.iter;
    broadcast = item.rs2[3];
    stream = item.rs2[9];
    cmd_cg.sample();
  endfunction
  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    verification_pkg::evidence::checks++;
    if (cmd_cg.get_coverage() != 100.0) `uvm_fatal("COV", "Fixed functional coverage incomplete")
  endfunction
endclass
