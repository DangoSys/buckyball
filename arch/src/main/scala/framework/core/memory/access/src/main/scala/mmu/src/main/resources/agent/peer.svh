class peer_driver extends uvm_component;
  `uvm_component_utils(peer_driver)
  virtual mmu_if vif;
  model reference;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual mmu_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "Walker PTE interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    forever begin
      @(negedge vif.clock);
      #1;
      vif.inputs[`MMU_IN_ACCESS_READY_OFFSET+:`MMU_IN_ACCESS_READY_WIDTH]=!vif.reset&&!reference.hold_access&&reference.cycle%5>=2;
      vif.inputs[`MMU_IN_RESP_READY_OFFSET+:`MMU_IN_RESP_READY_WIDTH]=!vif.reset&&!reference.hold_response&&reference.cycle%7>=3;
      vif.inputs[`MMU_IN_RESULT_VALID_OFFSET+:`MMU_IN_RESULT_VALID_WIDTH]=!vif.reset&&reference.pending_result&&!reference.hold_result&&reference.cycle>=reference.due;
      vif.inputs[`MMU_IN_RESULT_BITS_DATA_OFFSET+:`MMU_IN_RESULT_BITS_DATA_WIDTH]=reference.result_data;
      vif.inputs[`MMU_IN_RESULT_BITS_ERROR_OFFSET+:`MMU_IN_RESULT_BITS_ERROR_WIDTH]=reference.result_error;
    end
  endtask
endclass
