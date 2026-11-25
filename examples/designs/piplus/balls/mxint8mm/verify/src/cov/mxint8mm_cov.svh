// Semantic coverage of the commands: start values and destination, shape
// against the output tile, K blocks against the shared exponent row, layout,
// back-to-back commands, response backpressure, and the numeric events the
// reference reports (casegen/src/reference.rs F_*).
class mxint8mm_cov extends uvm_component;
  `uvm_component_utils(mxint8mm_cov)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, mxint8mm_cov) stim_imp;

  // {accumulate, bias, update}
  bit [2:0] cur_start;
  int unsigned cur_m;
  int unsigned cur_n;
  int unsigned cur_blocks;
  bit cur_tight;
  bit cur_offset;
  bit cur_chained;
  bit cur_stall;
  bit cur_edge;
  bit [8:0] cur_features;
  int unsigned cmd_count;

  covergroup cmd_cg;
    cp_start: coverpoint cur_start {
      bins zero = {3'b000, 3'b001};
      bins old_renamed = {3'b100};
      bins old_in_place = {3'b101};
      bins bias_renamed = {3'b010};
      bins bias_in_place = {3'b011};
      bins both_renamed = {3'b110};
      bins both_in_place = {3'b111};
    }
    cp_m: coverpoint cur_m {
      bins one = {1};
      bins partial = {[2 : `MXINT8MM_ROWS - 1]};
      bins tile = {`MXINT8MM_ROWS};
      bins tiles = {[`MXINT8MM_ROWS + 1 : 1023]} with (item % `MXINT8MM_ROWS == 0);
      bins tail = {[`MXINT8MM_ROWS + 1 : 1023]} with (item % `MXINT8MM_ROWS != 0);
    }
    cp_n: coverpoint cur_n {
      bins one_group = {4};
      bins tile = {`MXINT8MM_COLS};
      bins tiles = {[`MXINT8MM_COLS + 1 : 1023]} with (item % `MXINT8MM_COLS == 0);
      bins tail = {[`MXINT8MM_COLS + 1 : 1023]} with (item % `MXINT8MM_COLS != 0);
    }
    // One exponent bank row serves 16 blocks.
    cp_blocks: coverpoint cur_blocks {
      bins one = {1}; bins few = {[2 : 15]}; bins sixteen = {16}; bins more = {[17 : 1023]};
    }
    cp_pitch: coverpoint cur_tight {bins gap = {0}; bins tight = {1};}
    cp_a_base: coverpoint cur_offset {bins zero = {0}; bins offset = {1};}
    cp_chained: coverpoint cur_chained {bins first = {0}; bins chained = {1};}
    cp_stall: coverpoint cur_stall {bins held = {1};}
    cp_bank_end: coverpoint cur_edge {bins reached = {1};}
    cp_subnormal_term: coverpoint cur_features[0] {bins hit = {1};}
    cp_flushed_term: coverpoint cur_features[1] {bins hit = {1};}
    cp_huge_term: coverpoint cur_features[2] {bins hit = {1};}
    cp_cancelled_block: coverpoint cur_features[3] {bins hit = {1};}
    cp_code_min: coverpoint cur_features[4] {bins hit = {1};}
    cp_exponent_max: coverpoint cur_features[5] {bins hit = {1};}
    cp_exponent_min: coverpoint cur_features[6] {bins hit = {1};}
    cp_inexact_add: coverpoint cur_features[7] {bins hit = {1};}
    cp_special_start: coverpoint cur_features[8] {bins hit = {1};}
    start_blocks: cross cp_start, cp_blocks{
      ignore_bins rare = binsof (cp_blocks.sixteen) || binsof (cp_blocks.more);
    }
  endgroup

  function new(string name, uvm_component parent);
    super.new(name, parent);
    stim_imp = new("stim_imp", this);
    cmd_cg   = new();
  endfunction

  function void write_stim(bb_blink_cmd_item item);
    mxint8mm_cmd_item stim;
    int unsigned stride, a_base, pitch, c_base, entries;
    if (!$cast(stim, item)) `uvm_fatal("COV", "stim item is not mxint8mm_cmd_item")
    entries = 1 << `BB_BANK_ADDR_W;
    cur_m = stim.rs2[9:0];
    cur_n = stim.rs2[19:10];
    a_base = stim.rs2[33:20];
    pitch = stim.rs2[47:34];
    c_base = stim.rs2[61:48];
    cur_blocks = stim.iter / 32;
    stride = stim.iter / 16 + (stim.iter + 511) / 512;
    cur_start = {stim.rs2[62], stim.rs2[63], stim.update};
    cur_tight = pitch == cur_n / 4;
    cur_offset = a_base != 0;
    cur_chained = stim.step != 0;
    cur_stall = stim.stall;
    cur_edge     = a_base + cur_m * stride == entries || c_base + (cur_m - 1) * pitch + cur_n / 4 == entries;
    cur_features = stim.features;
    cmd_cg.sample();
    cmd_count++;
  endfunction

  function void check_phase(uvm_phase phase);
    verification_pkg::evidence::checks++;
    super.check_phase(phase);
    if (cmd_count == 0) `uvm_fatal("COV", "no cmd observed")
    if (cmd_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf(
                 "mxint8mm command coverage %0.2f%%: start %0.0f m %0.0f n %0.0f blocks %0.0f pitch %0.0f a_base %0.0f chained %0.0f start_blocks %0.0f",
                 cmd_cg.get_coverage(),
                 cmd_cg.cp_start.get_coverage(),
                 cmd_cg.cp_m.get_coverage(),
                 cmd_cg.cp_n.get_coverage(),
                 cmd_cg.cp_blocks.get_coverage(),
                 cmd_cg.cp_pitch.get_coverage(),
                 cmd_cg.cp_a_base.get_coverage(),
                 cmd_cg.cp_chained.get_coverage(),
                 cmd_cg.start_blocks.get_coverage()
                 ))
  endfunction
endclass
