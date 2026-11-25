class maxpool_cov extends uvm_component;
  `uvm_component_utils(maxpool_cov)

  uvm_analysis_imp_cmd #(bb_blink_cmd_item, maxpool_cov) cmd_imp;
  uvm_analysis_imp_stim #(bb_blink_cmd_item, maxpool_cov) stim_imp;

  int unsigned cur_kernel;
  int unsigned cur_stride;
  int unsigned cur_padding;
  int unsigned cur_groups;
  int cmd_count;
  int unsigned cur_k;
  int unsigned cur_key;
  bit cur_odd;
  bit cur_tail;
  bit cur_full;
  bit cur_large_k;

  covergroup cmd_cg;
    coverpoint cur_kernel {bins k2 = {2}; bins k3 = {3};}
    coverpoint cur_stride {bins s1 = {1}; bins s2 = {2};}
    coverpoint cur_padding {bins p0 = {0}; bins p1 = {1};}
    coverpoint cur_groups {bins one = {1}; bins many = {[2 : 24]};}
  endgroup

  covergroup topk_cg;
    coverpoint cur_groups {bins one = {1}; bins many = {[2 : 24]};}
    coverpoint cur_k {bins one = {1}; bins many = {[2 : 512]};}
    coverpoint cur_odd {bins even_k = {0}; bins odd_k = {1};}
    coverpoint cur_tail {bins full_input = {0}; bins partial_input = {1};}
    coverpoint cur_full {bins partial_k = {0}; bins all_keys = {1};}
    coverpoint cur_large_k {bins within_255 = {0}; bins above_255 = {1};}
  endgroup

  covergroup key_cg;
    coverpoint cur_key {
      bins zero = {0};
      bins low = {[1 : 32'h7fffffff]};
      bins sign_edge = {32'h80000000};
      bins high = {[32'h80000001 : 32'hfffffffe]};
      bins maximum = {32'hffffffff};
    }
  endgroup

  function new(string name, uvm_component parent);
    super.new(name, parent);
    cmd_imp  = new("cmd_imp", this);
    stim_imp = new("stim_imp", this);
    cmd_cg   = new();
    topk_cg  = new();
    key_cg   = new();
  endfunction

  function void write_stim(bb_blink_cmd_item item);
    maxpool_cmd_item stim;
    if (!$cast(stim, item)) `uvm_fatal("COV", "invalid MaxPool stimulus type")
    if (stim.is_topk)
      for (int unsigned i = 0; i < stim.rs2[31:0]; i++) begin
        cur_key = stim.input_words[i/4][(i%4)*32+:32];
        key_cg.sample();
      end
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    if (item.funct == TOPK_THREAD_FUNCT7[6:0]) begin
      cur_groups = int'(item.op1_col);
      cur_k = int'(item.rs2[47:32]);
      cur_odd = cur_k[0];
      cur_tail = item.rs2[31:0] != item.iter * item.op1_col * 4;
      cur_full = item.rs2[31:0] == cur_k;
      cur_large_k = cur_k > 255;
      topk_cg.sample();
      cmd_count++;
      return;
    end
    if (item.funct != MAXPOOL_THREAD_FUNCT7[6:0])
      `uvm_fatal("COV", $sformatf("unexpected funct %0d", item.funct))
    cur_kernel  = int'(item.rs2[11:8]);
    cur_stride  = int'(item.rs2[15:12]);
    cur_padding = int'(item.rs2[19:16]);
    cur_groups  = int'(item.op1_col);
    cmd_count++;
    cmd_cg.sample();
  endfunction

  function void check_phase(uvm_phase phase);
    verification_pkg::evidence::checks++;
    super.check_phase(phase);
    if (cmd_count == 0) `uvm_fatal("COV", "no cmd observed")
    if (cmd_cg.get_coverage() != 100.0) begin
      `uvm_fatal(
          "COV",
          $sformatf(
              "maxpool_cov coverage incomplete: %0.2f%% (need kernel {2,3}, stride {1,2}, pad {0,1}, groups {1,>1})",
              cmd_cg.get_coverage()))
    end
    if (topk_cg.get_coverage() != 100.0 || key_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf(
                 "Top-K coverage incomplete: command=%0.2f key=%0.2f",
                 topk_cg.get_coverage(),
                 key_cg.get_coverage()
                 ))
    `uvm_info("COV", $sformatf(
              "MaxPool=%0.2f%% Top-K=%0.2f%% U32 keys=%0.2f%%",
              cmd_cg.get_coverage(),
              topk_cg.get_coverage(),
              key_cg.get_coverage()
              ), UVM_LOW)
  endfunction
endclass
