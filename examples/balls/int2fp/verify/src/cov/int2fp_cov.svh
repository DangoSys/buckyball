class int2fp_cov extends uvm_component;
  `uvm_component_utils(int2fp_cov)

  uvm_analysis_imp_cmd #(bb_blink_cmd_item, int2fp_cov) cmd_imp;

  int cmd_count;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    cmd_imp = new("cmd_imp", this);
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    cmd_count++;
  endfunction

  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    if (cmd_count == 0) `uvm_fatal("COV", "no cmd observed")
  endfunction
endclass
