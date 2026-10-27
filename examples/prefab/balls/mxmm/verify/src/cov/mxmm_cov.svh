class mxmm_cov extends uvm_component;
  `uvm_component_utils(mxmm_cov)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, mxmm_cov) stim_imp;

  int unsigned cur_format;
  bit [1:0] cur_chain;
  int unsigned cur_m;
  int unsigned cur_panels;
  int unsigned cmd_count;

  covergroup cmd_cg;
    coverpoint cur_format {bins mxfp8 = {0}; bins window = {1}; bins fma32 = {2}; bins f32 = {3};}
    // {first, last}
    coverpoint cur_chain {
      bins single = {2'b11}; bins open = {2'b10}; bins middle = {2'b00}; bins close = {2'b01};
    }
    coverpoint cur_m {bins vector = {1}; bins tile = {16}; bins tiles = {[32 : 4095]};}
    // Panels beyond the seven contexts need a second batch.
    coverpoint cur_panels {
      bins one = {1}; bins batch = {[2 : 7]}; bins batches = {[8 : 4095]};
    }
    format_chain: cross cur_format, cur_chain{
      ignore_bins fused_chain = binsof (cur_format) intersect {2} && !binsof (cur_chain) intersect {
        2'b11
      };
      ignore_bins f32_middle = binsof (cur_format) intersect {3} && binsof (cur_chain) intersect {
        2'b00
      };
    }
  endgroup

  function new(string name, uvm_component parent);
    super.new(name, parent);
    stim_imp = new("stim_imp", this);
    cmd_cg   = new();
  endfunction

  function void write_stim(bb_blink_cmd_item item);
    mxmm_cmd_item stim;
    int unsigned  n;
    if (!$cast(stim, item)) `uvm_fatal("COV", "stim item is not mxmm_cmd_item")
    cur_format = stim.format;
    cur_chain  = {stim.rs2[24], stim.rs2[25]};
    cur_m      = stim.rs2[11:0];
    n          = stim.rs2[23:12];
    cur_panels = (cur_m == 1 ? 1 : cur_m / 16) * (n / 16);
    cmd_cg.sample();
    cmd_count++;
  endfunction

  function void check_phase(uvm_phase phase);
    verification_pkg::evidence::checks++;
    super.check_phase(phase);
    if (cmd_count == 0) `uvm_fatal("COV", "no cmd observed")
    if (cmd_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf("mxmm command coverage %0.2f%%", cmd_cg.get_coverage()))
  endfunction
endclass
