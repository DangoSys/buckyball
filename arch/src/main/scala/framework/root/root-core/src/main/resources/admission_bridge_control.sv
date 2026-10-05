`include "admission_bridge_config.svh"
interface admission_bridge_control_if (
    input logic clock
);
  logic reset, boot_allocation, allocation_valid;
  logic fault_valid, unbound_fault_valid;
  logic [`AB_FAULT_WIDTH-1:0] fault_bits, unbound_fault_bits;
  logic [`AB_ROB_BITS-1:0] allocation_id, lookup_id[3];
  logic [`AB_ROB_ENTRIES-1:0] retired;
  logic lookup_valid[3];
  logic [`AB_SNAP_WIDTH-1:0] lookup_bits[3];
  clocking sample @(posedge clock);
    default input #1step;
    input fault_valid, unbound_fault_valid, fault_bits, unbound_fault_bits;
    input reset, allocation_valid, allocation_id, retired, lookup_id, lookup_valid, lookup_bits;
  endclocking
endinterface
