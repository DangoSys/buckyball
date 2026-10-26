class env extends uvm_env;
  `uvm_component_utils(env)
  virtual dma_if vif;
  covergroup behavior with function sample (
      longint unsigned base,
      int length,
      int error,
      bit bus_error,
      bit two_dimensional,
      bit cancellation
  );
    option.per_instance = 1;
    alignment: coverpoint (base % 16) {bins aligned = {0}; bins unaligned = {[1 : 15]};}
    lengths: coverpoint length {
      bins invalid = {0}; bins short_transfer = {[1 : 64]}; bins long_transfer = {[65 : $]};
    }
    faults: coverpoint error {bins success = {0}; bins bus = {2}; bins shape = {7};}
    backend_errors: coverpoint bus_error;
    dimensionality: coverpoint two_dimensional;
    reset_cancellation: coverpoint cancellation;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    behavior = new();
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual dma_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "DMA interface is required")
  endfunction
  int cases = 0, beats = 0, bursts = 0, checks = 0;
  function automatic bit [127:0] data_at(longint unsigned address);
    bit [127:0] value;
    for (int i = 0; i < 16; ++i) value[i*8+:8] = 8'(dma_ref_byte(address + i));
    return value;
  endfunction
  task automatic verify_contract(bit good, string reason);
    verification_pkg::evidence::checks++;
    checks++;
    if (!good) `uvm_fatal("READ_DMA", reason)
  endtask

  `include "seq/scenario.svh"
endclass
