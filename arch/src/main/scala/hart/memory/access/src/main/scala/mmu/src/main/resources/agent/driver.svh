class driver extends uvm_driver #(request_item);
  `uvm_component_utils(driver)
  virtual mmu_if vif;
  request_item pending;
  int mode = 0;
  longint unsigned root = 0;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual mmu_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "Walker input interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    forever begin
      @(negedge vif.clock);
      if (pending == null) seq_item_port.try_next_item(pending);
      vif.inputs[`MMU_IN_FLUSH_OFFSET+:`MMU_IN_FLUSH_WIDTH] = 0;
      vif.inputs[`MMU_IN_REQ_VALID_OFFSET+:`MMU_IN_REQ_VALID_WIDTH] = 0;
      if (pending != null) begin
        if (pending.kind == 0) begin
          mode = pending.mode;
          root = pending.root;
          vif.inputs[`MMU_REQ_OFFSET+:`MMU_REQ_BITS] = pending.payload;
          vif.inputs[`MMU_IN_REQ_VALID_OFFSET+:`MMU_IN_REQ_VALID_WIDTH] = 1;
        end else if (pending.kind == 1) vif.inputs[`MMU_IN_FLUSH_OFFSET+:`MMU_IN_FLUSH_WIDTH] = 1;
        else begin
          vif.inputs[`MMU_LOOKUP_OFFSET+:`MMU_LOOKUP_BITS] = pending.payload;
          vif.inputs[`MMU_IN_LOOKUP_CONFIG_MODE_OFFSET+:`MMU_IN_LOOKUP_CONFIG_MODE_WIDTH]=pending.mode;
          vif.inputs[`MMU_IN_LOOKUP_CONFIG_ROOTPPN_OFFSET+:`MMU_IN_LOOKUP_CONFIG_ROOTPPN_WIDTH]=pending.root;
        end
      end
      vif.inputs[`MMU_IN_CONFIG_MODE_OFFSET+:`MMU_IN_CONFIG_MODE_WIDTH] = mode;
      vif.inputs[`MMU_IN_CONFIG_ROOTPPN_OFFSET+:`MMU_IN_CONFIG_ROOTPPN_WIDTH] = root;
      @(posedge vif.clock);
      if (!vif.reset && pending != null && (pending.kind != 0 || `MO(vif.outputs, REQ_READY))) begin
        if (pending.kind == 0 && pending.change_inputs) begin
          mode = pending.next_mode;
          root = pending.next_root;
        end
        #1;
        seq_item_port.item_done();
        pending = null;
      end
    end
  endtask
endclass
