class toint8_cov extends uvm_component;
  `uvm_component_utils(toint8_cov)

  uvm_analysis_imp_cmd #(bb_blink_cmd_item, toint8_cov) cmd_imp;

  bit [6:0] cur_funct;
  int unsigned cur_groups;

  covergroup cmd_cg;
    coverpoint cur_funct {
      bins f32 = {QUANT_F32_TO_I8_THREAD_FUNCT7[6:0]};
      bins i32 = {QUANT_I32_TO_I8_THREAD_FUNCT7[6:0]};
    }
    coverpoint cur_groups {bins single = {1}; bins multi = {[2 : $]};}
  endgroup

  int cmd_count;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    cmd_imp = new("cmd_imp", this);
    cmd_cg  = new();
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    cur_funct  = item.funct;
    cur_groups = item.op1_col;
    cmd_count++;
    cmd_cg.sample();
  endfunction

  function void check_phase(uvm_phase phase);
    verification_pkg::evidence::checks++;
    super.check_phase(phase);
    if (cmd_count == 0) `uvm_fatal("COV", "no cmd observed")
    if (cmd_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf("toint8_cov %0.2f%%", cmd_cg.get_coverage()))
  endfunction
endclass
