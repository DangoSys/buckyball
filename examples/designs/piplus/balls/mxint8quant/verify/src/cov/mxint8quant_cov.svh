// Semantic coverage of the commands driven: shapes and layouts, the exponent
// and code values of the reference output and the rounding events of the
// reference. Every bin must be hit.
class mxint8quant_cov extends uvm_component;
  `uvm_component_utils(mxint8quant_cov)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, mxint8quant_cov) stim_imp;

  int unsigned cur_k;
  int unsigned cur_rows;
  bit cur_columns;
  bit cur_gap;
  int unsigned cur_exponent_rows;
  bit cur_in_base;
  bit cur_out_base;
  bit cur_hold;
  byte unsigned cur_exponent;
  byte cur_code;
  int unsigned cur_event;
  int unsigned cmd_count;

  covergroup cmd_cg;
    cp_k: coverpoint cur_k {
      bins one_block = {32};
      bins blocks = {[64 : 480]};
      bins sixteen_blocks = {512};
      bins exponent_rows = {[544 : 4064]};
      bins bank = {4096};
    }
    cp_rows: coverpoint cur_rows {
      bins one = {1};
      bins few = {[2 : 3]};
      bins four = {4};
      bins some = {[5 : 15]};
      bins many = {[16 : 64]};
    }
    cp_columns: coverpoint cur_columns;
    cp_gap: coverpoint cur_gap;
    cp_exponent_rows: coverpoint cur_exponent_rows {
      bins one = {1}; bins two = {2}; bins many = {[3 : 8]};
    }
    cp_in_base: coverpoint cur_in_base;
    cp_out_base: coverpoint cur_out_base;
    cp_hold: coverpoint cur_hold;
    x_layout: cross cp_columns, cp_gap;
    x_exponent_rows: cross cp_columns, cp_exponent_rows{
      // The K input rows of a column fit in the bank only for K <= 1024.
      ignore_bins columns_many = binsof (cp_columns) intersect {1} &&
                                 binsof (cp_exponent_rows.many);
    }
  endgroup

  covergroup exponent_cg;
    coverpoint cur_exponent {
      bins clamped = {0};
      bins low = {[1 : 117]};
      bins unit = {[118 : 136]};
      bins high = {[137 : 247]};
      bins max_finite = {248};
    }
  endgroup

  covergroup code_cg;
    coverpoint cur_code {
      bins negative_clamp = {-127};
      bins negative = {[-126 : -1]};
      bins zero = {0};
      bins positive = {[1 : 126]};
      bins positive_clamp = {127};
    }
  endgroup

  covergroup event_cg;
    coverpoint cur_event {
      bins zero_block = {0};
      bins subnormal_block = {1};
      bins clamped_exponent = {2};
      bins tie_down = {3};
      bins tie_up = {4};
      bins saturate = {5};
      bins flush = {6};
      bins negative_zero = {7};
      bins subnormal_element = {8};
      bins max_finite = {9};
    }
  endgroup

  function new(string name, uvm_component parent);
    super.new(name, parent);
    stim_imp    = new("stim_imp", this);
    cmd_cg      = new();
    exponent_cg = new();
    code_cg     = new();
    event_cg    = new();
  endfunction

  function void write_stim(bb_blink_cmd_item item);
    mxint8quant_cmd_item stim;
    int unsigned width;
    if (!$cast(stim, item)) `uvm_fatal("COV", "stim item is not mxint8quant_cmd_item")
    cur_k = stim.rs2[15:0];
    cur_rows = stim.iter;
    cur_columns = stim.rs2[44];
    width = cur_columns ? cur_rows / 4 : cur_k / 4;
    cur_gap = stim.rs2[58:45] > width;
    cur_exponent_rows = (cur_k + 511) / 512;
    cur_in_base = stim.rs2[29:16] != 0;
    cur_out_base = stim.rs2[43:30] != 0;
    cur_hold = stim.hold != 0;
    cmd_cg.sample();
    foreach (stim.write_data[i]) begin
      for (int b = 0; b < 16; b++) begin
        if (stim.write_kind[i] == 0) begin
          cur_code = stim.write_data[i][b*8+:8];
          code_cg.sample();
        end else if (b < stim.write_kind[i]) begin
          cur_exponent = stim.write_data[i][b*8+:8];
          exponent_cg.sample();
        end
      end
    end
    foreach (stim.events[e])
    if (stim.events[e] != 0) begin
      cur_event = e;
      event_cg.sample();
    end
    cmd_count++;
  endfunction

  function void check_phase(uvm_phase phase);
    verification_pkg::evidence::checks++;
    super.check_phase(phase);
    if (cmd_count == 0) `uvm_fatal("COV", "no cmd observed")
    if (cmd_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf("mxint8quant command coverage %0.2f%%", cmd_cg.get_coverage()))
    if (exponent_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf(
                 "mxint8quant exponent coverage %0.2f%%", exponent_cg.get_coverage()))
    if (code_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf("mxint8quant code coverage %0.2f%%", code_cg.get_coverage()))
    if (event_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf("mxint8quant event coverage %0.2f%%", event_cg.get_coverage()))
  endfunction
endclass
