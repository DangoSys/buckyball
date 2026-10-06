`ifndef RNF_TB
`define RNF_TB rnf_tb
`endif
interface chi_rnf_if (
    input logic clock
);
  logic reset;
  logic access_valid, access_ready, access_write;
  logic [43:0] access_addr;
  logic [63:0] access_data;
  logic [7:0] access_mask;
  logic [3:0] access_atomic;
  logic access_atomic_word;
  logic result_valid, result_ready, result_error;
  logic [63:0] result_data;
  logic req_valid, req_ready, rsp_valid, rsp_ready, dat_valid, dat_ready;
  logic rx_rsp_valid, rx_rsp_ready, rx_dat_valid, rx_dat_ready, snp_valid, snp_ready;
  logic [136:0] req;
  logic [70:0] rsp, rx_rsp;
  logic [388:0] dat, rx_dat;
  logic [93:0] snp;
  logic [31:0] outstanding, hits, misses;
  // Statistics only: compress a prefix of repeated legal hits/misses.
  // Cache contents, coherence state, transactions, and credits are never seeded.
  task seed_counter(bit bank, bit miss, int unsigned value);
    if (bank) begin
      if (miss) $deposit($root.`RNF_TB.dut.caches_1.misses, value);
      else $deposit($root.`RNF_TB.dut.caches_1.hits, value);
    end else begin
      if (miss) $deposit($root.`RNF_TB.dut.caches_0.misses, value);
      else $deposit($root.`RNF_TB.dut.caches_0.hits, value);
    end
  endtask
  function int unsigned counter_value(bit bank, bit miss);
    if (bank) return miss ? $root.`RNF_TB.dut.caches_1.misses : $root.`RNF_TB.dut.caches_1.hits;
    return miss ? $root.`RNF_TB.dut.caches_0.misses : $root.`RNF_TB.dut.caches_0.hits;
  endfunction
  assert property (@(posedge clock) disable iff (reset)
    access_valid && !access_ready |=> access_valid && $stable(
      {access_addr, access_write, access_data, access_mask, access_atomic, access_atomic_word}
  ));
  assert property (@(posedge clock) disable iff (reset)
    req_valid && !req_ready |=> req_valid && $stable(
      req
  ));
  assert property (@(posedge clock) disable iff (reset)
    rsp_valid && !rsp_ready |=> rsp_valid && $stable(
      rsp
  ));
  assert property (@(posedge clock) disable iff (reset)
    dat_valid && !dat_ready |=> dat_valid && $stable(
      dat
  ));
  assert property (@(posedge clock) disable iff (reset)
    result_valid && !result_ready |=> result_valid && $stable(
      {result_error, result_data}
  ));
endinterface
