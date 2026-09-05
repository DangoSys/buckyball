class toint8_cov extends uvm_component;
  `uvm_component_utils(toint8_cov)

  uvm_analysis_imp_cmd #(bb_blink_cmd_item, toint8_cov) cmd_imp;

  bit [6:0] cur_funct7;

  covergroup cmd_cg;
    coverpoint cur_funct7 {
      bins f32 = {QUANT_F32_TO_I8_CORE_FUNCT7[6:0]}; bins i32 = {QUANT_I32_TO_I8_CORE_FUNCT7[6:0]};
    }
  endgroup

  int cmd_count;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    cmd_imp = new("cmd_imp", this);
    cmd_cg  = new();
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    cur_funct7 = item.funct7;
    cmd_count++;
    cmd_cg.sample();
  endfunction

  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    if (cmd_count == 0) `uvm_fatal("COV", "no cmd observed")
    if (cmd_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf("toint8_cov %0.2f%%", cmd_cg.get_coverage()))
  endfunction
endclass
