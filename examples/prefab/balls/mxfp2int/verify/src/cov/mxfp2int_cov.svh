class mxfp2int_cov extends uvm_component;
  `uvm_component_utils(mxfp2int_cov)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, mxfp2int_cov) stim_imp;

  int unsigned cur_iter;
  int unsigned cur_scale;
  int unsigned cmd_count;

  covergroup cmd_cg;
    coverpoint cur_iter {
      bins one = {1};
      bins row = {[2 : 16]};
      bins rows = {[17 : 64]};
      bins many = {[65 : 511]};
      bins capacity = {512};
    }
  endgroup

  covergroup scale_cg;
    coverpoint cur_scale {
      bins zero = {0};
      bins shift_past_31 = {[1 : 95]};
      bins right = {[96 : 126]};
      bins identity = {127};
      bins left = {[128 : 134]};
      bins saturate = {[135 : 254]};
      bins top = {255};
    }
  endgroup

  function new(string name, uvm_component parent);
    super.new(name, parent);
    stim_imp = new("stim_imp", this);
    cmd_cg   = new();
    scale_cg = new();
  endfunction

  function void write_stim(bb_blink_cmd_item item);
    mxfp2int_cmd_item stim;
    if (!$cast(stim, item)) `uvm_fatal("COV", "stim item is not mxfp2int_cmd_item")
    cur_iter = stim.iter;
    cmd_cg.sample();
    foreach (stim.preload_data[i]) begin
      if (stim.preload_bank[i] != stim.op2_bank) continue;
      for (int b = 0; b < 16; b++) begin
        if (stim.preload_addr[i] * 16 + b >= stim.iter) break;
        cur_scale = stim.preload_data[i][b*8+:8];
        scale_cg.sample();
      end
    end
    cmd_count++;
  endfunction

  function void check_phase(uvm_phase phase);
    verification_pkg::evidence::checks++;
    super.check_phase(phase);
    if (cmd_count == 0) `uvm_fatal("COV", "no cmd observed")
    if (cmd_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf("mxfp2int iter coverage %0.2f%%", cmd_cg.get_coverage()))
    if (scale_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf("mxfp2int scale coverage %0.2f%%", scale_cg.get_coverage()))
  endfunction
endclass
