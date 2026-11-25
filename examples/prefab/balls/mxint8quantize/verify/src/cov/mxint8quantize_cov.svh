class mxint8quantize_cov extends uvm_component;
  `uvm_component_utils(mxint8quantize_cov)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, mxint8quantize_cov) stim_imp;

  int unsigned cur_k;
  int unsigned cur_rows;
  int unsigned cur_in_base;
  int unsigned cur_out_base;
  bit cur_in_full;
  bit cur_out_full;
  int unsigned cur_exponent;
  bit [7:0] cur_code;
  int unsigned cur_event;
  int unsigned cmd_count;

  covergroup cmd_cg;
    coverpoint cur_k {
      bins one_block = {32};
      bins one_exponent_row = {[64 : 480]};
      bins full_exponent_row = {512};
      bins tail_exponent_row = {[544 : 4064]};
      bins capacity = {4096};
    }
    coverpoint (cur_k % 512 == 0) {bins tail = {0}; bins whole = {1};}
    coverpoint cur_rows {bins one = {1}; bins few = {[2 : 8]}; bins many = {[9 : 1024]};}
    coverpoint cur_in_base {bins zero = {0}; bins offset = {[1 : 1023]};}
    coverpoint cur_out_base {bins zero = {0}; bins offset = {[1 : 1023]};}
    coverpoint cur_in_full {bins partial = {0}; bins full = {1};}
    coverpoint cur_out_full {bins partial = {0}; bins full = {1};}
  endgroup

  // Exponent byte e + 127: 0 is the clamp at -127, 127 an all-zero block or
  // F = 133, 248 the largest finite maximum.
  covergroup exponent_cg;
    coverpoint cur_exponent {
      bins clamped = {0};
      bins low = {[1 : 126]};
      bins unit = {127};
      bins high = {[128 : 247]};
      bins top = {248};
    }
  endgroup

  covergroup code_cg;
    coverpoint cur_code {
      bins zero = {8'h00};
      bins pos_small = {[8'h01 : 8'h3f]};
      bins pos_large = {[8'h40 : 8'h7e]};
      bins pos_max = {8'h7f};
      bins neg_max = {8'h81};
      bins neg_large = {[8'h82 : 8'hc0]};
      bins neg_small = {[8'hc1 : 8'hff]};
      illegal_bins minus_128 = {8'h80};
    }
  endgroup

  // Numeric events of the reference model (casegen.rs, ZERO_BLOCK ..
  // NEGATIVE_CODE): every one must occur in some command.
  covergroup event_cg;
    coverpoint cur_event {
      bins zero_block = {0};
      bins subnormal_max = {1};
      bins clamped_exponent = {2};
      bins top_exponent = {3};
      bins tie_down = {4};
      bins tie_up = {5};
      bins saturate = {6};
      bins flush = {7};
      bins negative_zero = {8};
      bins subnormal_input = {9};
      bins negative_code = {10};
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
    mxint8quantize_cmd_item stim;
    int unsigned stride;
    int unsigned offset;
    int unsigned blocks;
    if (!$cast(stim, item)) `uvm_fatal("COV", "stim item is not mxint8quantize_cmd_item")
    cur_k        = stim.rs2[15:0];
    cur_in_base  = stim.rs2[29:16];
    cur_out_base = stim.rs2[43:30];
    cur_rows     = stim.iter;
    stride       = cur_k / 16 + (cur_k + 511) / 512;
    cur_in_full  = cur_in_base + cur_rows * cur_k / 4 == mxint8quantize_bank_entries();
    cur_out_full = cur_out_base + cur_rows * stride == mxint8quantize_bank_entries();
    cmd_cg.sample();
    foreach (stim.write_data[i]) begin
      offset = (stim.write_addr[i] - cur_out_base) % stride;
      if (offset < cur_k / 16) begin
        for (int b = 0; b < 16; b++) begin
          cur_code = stim.write_data[i][b*8+:8];
          code_cg.sample();
        end
      end else begin
        blocks = cur_k / 32 - 16 * (offset - cur_k / 16);
        for (int b = 0; b < 16 && b < blocks; b++) begin
          cur_exponent = stim.write_data[i][b*8+:8];
          exponent_cg.sample();
        end
      end
    end
    foreach (stim.events[e]) begin
      if (stim.events[e] == 0) continue;
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
      `uvm_fatal("COV", $sformatf("mxint8quantize command coverage %0.2f%%", cmd_cg.get_coverage()))
    if (exponent_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf(
                 "mxint8quantize exponent coverage %0.2f%%", exponent_cg.get_coverage()))
    if (code_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf("mxint8quantize code coverage %0.2f%%", code_cg.get_coverage()))
    if (event_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf("mxint8quantize event coverage %0.2f%%", event_cg.get_coverage()))
  endfunction
endclass
