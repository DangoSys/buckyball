class int2fp_cov extends uvm_component;
  `uvm_component_utils(int2fp_cov)

  uvm_analysis_imp_cmd #(bb_blink_cmd_item, int2fp_cov) cmd_imp;

  bit cur_relu;
  int unsigned cur_iter;

  covergroup cmd_cg;
    coverpoint cur_relu {bins off = {0}; bins on = {1};}
    coverpoint cur_iter {bins i4 = {4}; bins i8 = {8}; bins i16 = {16};}
  endgroup

  int cmd_count;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    cmd_imp = new("cmd_imp", this);
    cmd_cg  = new();
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    if (item.funct7 != INT32_TO_FP32_CORE_FUNCT7[6:0])
      `uvm_fatal("COV", $sformatf("unexpected funct7 %0d", item.funct7))
    cur_relu = item.rs2[0];
    cur_iter = item.iter;
    cmd_count++;
    cmd_cg.sample();
  endfunction

  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    if (cmd_count == 0) `uvm_fatal("COV", "no cmd observed")
    if (cmd_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf("int2fp_cov %0.2f%%", cmd_cg.get_coverage()))
  endfunction
endclass
