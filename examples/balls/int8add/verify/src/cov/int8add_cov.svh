class int8add_cov extends uvm_component;
  `uvm_component_utils(int8add_cov)

  uvm_analysis_imp_cmd #(bb_blink_cmd_item, int8add_cov) cmd_imp;

  bit [6:0] cur_funct7;

  covergroup cmd_cg;
    coverpoint cur_funct7 {
      bins add = {INT8ADD_CORE_FUNCT7[6:0]}; bins relu = {INT8ADD_RELU_CORE_FUNCT7[6:0]};
    }
  endgroup

  int cmd_count;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    cmd_imp = new("cmd_imp", this);
    cmd_cg  = new();
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    if (item.funct7 != INT8ADD_CORE_FUNCT7[6:0] && item.funct7 != INT8ADD_RELU_CORE_FUNCT7[6:0])
      `uvm_fatal("COV", $sformatf("unexpected funct7 %0d", item.funct7))
    cur_funct7 = item.funct7;
    cmd_count++;
    cmd_cg.sample();
  endfunction

  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    if (cmd_count == 0) `uvm_fatal("COV", "no cmd observed")
    if (cmd_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf("int8add_cov %0.2f%%", cmd_cg.get_coverage()))
  endfunction
endclass
