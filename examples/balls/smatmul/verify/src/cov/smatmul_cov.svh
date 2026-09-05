class smatmul_cov extends uvm_component;
  `uvm_component_utils(smatmul_cov)

  uvm_analysis_imp_cmd #(bb_blink_cmd_item, smatmul_cov) cmd_imp;
  bit is_bias, is_first, is_cont;

  covergroup cmd_cg;
    coverpoint is_bias {bins bias = {1}; bins os = {0};}
    coverpoint is_first {bins first = {1}; bins cont = {0};}
  endgroup

  int cmd_count;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    cmd_imp = new("cmd_imp", this);
    cmd_cg  = new();
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    is_bias  = (item.funct7 == SMATMUL_BIAS_CORE_FUNCT7[6:0]);
    is_first = is_bias ? 0 : item.rs2[24];
    cmd_count++;
    cmd_cg.sample();
  endfunction

  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    if (cmd_count == 0) `uvm_fatal("COV", "no cmd observed")
    if (cmd_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf("smatmul_cov %0.2f%%", cmd_cg.get_coverage()))
  endfunction
endclass
