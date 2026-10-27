class mxquant_cov extends uvm_component;
  `uvm_component_utils(mxquant_cov)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, mxquant_cov) stim_imp;

  int unsigned cur_count;
  int unsigned cur_scale;
  bit [6:0] cur_magnitude;
  bit cur_sign;
  int unsigned cmd_count;

  covergroup cmd_cg;
    coverpoint cur_count {
      bins one_block = {32};
      bins blocks = {[33 : 1024]};
      bins many = {[1025 : 2047]};
      bins capacity = {2048};
    }
  endgroup

  covergroup scale_cg;
    coverpoint cur_scale {
      bins clamped = {0};
      bins low = {[1 : 118]};
      bins unit = {[119 : 135]};
      bins high = {[136 : 246]};
    }
  endgroup

  covergroup code_cg;
    coverpoint cur_magnitude {
      bins zero = {0}; bins subnormal = {[1 : 7]}; bins normal = {[8 : 125]}; bins clipped = {126};
    }
    coverpoint cur_sign;
  endgroup

  function new(string name, uvm_component parent);
    super.new(name, parent);
    stim_imp = new("stim_imp", this);
    cmd_cg   = new();
    scale_cg = new();
    code_cg  = new();
  endfunction

  function void write_stim(bb_blink_cmd_item item);
    mxquant_cmd_item stim;
    if (!$cast(stim, item)) `uvm_fatal("COV", "stim item is not mxquant_cmd_item")
    cur_count = stim.iter;
    cmd_cg.sample();
    foreach (stim.write_data[i]) begin
      for (int b = 0; b < 16; b++) begin
        if (!stim.write_mask[i][b]) continue;
        if (stim.write_mask[i] == 16'hFFFF) begin
          {cur_sign, cur_magnitude} = stim.write_data[i][b*8+:8];
          code_cg.sample();
        end else begin
          cur_scale = stim.write_data[i][b*8+:8];
          scale_cg.sample();
        end
      end
    end
    cmd_count++;
  endfunction

  function void check_phase(uvm_phase phase);
    verification_pkg::evidence::checks++;
    super.check_phase(phase);
    if (cmd_count == 0) `uvm_fatal("COV", "no cmd observed")
    if (cmd_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf("mxquant count coverage %0.2f%%", cmd_cg.get_coverage()))
    if (scale_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf("mxquant scale coverage %0.2f%%", scale_cg.get_coverage()))
    if (code_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf("mxquant code coverage %0.2f%%", code_cg.get_coverage()))
  endfunction
endclass
