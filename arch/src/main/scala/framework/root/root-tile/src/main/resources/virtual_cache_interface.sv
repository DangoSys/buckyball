`include "virtual_cache_config.svh"
interface virtual_cache_if (
    input logic clock
);
  logic reset, active, conflict_trace;
  int conflict_wb, conflict_dbid, conflict_beats, conflict_ack;
  logic [`VM_OUTSTANDING_BITS-1:0] outstanding;
  logic translation_valid, physical_valid, cache_valid, cache_pte, eviction_valid;
  logic [`VM_TRANSLATION_WIDTH-1:0] translation_bits;
  logic [`VM_PHYSICAL_WIDTH-1:0] physical_bits;
  logic [`VM_CACHE_WIDTH-1:0] cache_bits;
  logic [`VM_CACHE_ADDR_WIDTH-1:0] eviction_addr;
  clocking sample @(posedge clock);
    default input #1step;
    input reset,active,outstanding,translation_valid,physical_valid,cache_valid,cache_pte,eviction_valid;
    input translation_bits, physical_bits, cache_bits, eviction_addr;
  endclocking
endinterface
