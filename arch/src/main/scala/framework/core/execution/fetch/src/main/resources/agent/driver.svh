class driver extends uvm_driver #(control_item);
  `uvm_component_utils(driver)
  virtual fetch_if vif;
  program_image image;
  control_item pending;
  longint unsigned reset_vector = 0, satp = 0;
  int privilege = 3;
  bit sum = 0, mxr = 0;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual fetch_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "Fetch control interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    forever begin
      @(negedge vif.clock);
      if (pending == null) seq_item_port.try_next_item(pending);
      vif.inputs[`FETCH_IN_REDIRECT_VALID_OFFSET+:`FETCH_IN_REDIRECT_VALID_WIDTH] = 0;
      vif.inputs[`FETCH_IN_FLUSH_OFFSET+:`FETCH_IN_FLUSH_WIDTH] = 0;
      vif.inputs[`FETCH_IN_INVALIDATE_OFFSET+:`FETCH_IN_INVALIDATE_WIDTH] = 0;
      if (pending != null) begin
        case (pending.kind)
          REDIRECT: begin
            vif.inputs[`FETCH_IN_REDIRECT_VALID_OFFSET+:`FETCH_IN_REDIRECT_VALID_WIDTH] = 1;
            vif.inputs[`FETCH_IN_REDIRECT_BITS_OFFSET+:`FETCH_IN_REDIRECT_BITS_WIDTH]=pending.target;
          end
          FENCE: begin
            vif.inputs[`FETCH_IN_REDIRECT_VALID_OFFSET+:`FETCH_IN_REDIRECT_VALID_WIDTH] = 1;
            vif.inputs[`FETCH_IN_REDIRECT_BITS_OFFSET+:`FETCH_IN_REDIRECT_BITS_WIDTH]=pending.target;
            vif.inputs[`FETCH_IN_FLUSH_OFFSET+:`FETCH_IN_FLUSH_WIDTH] = 1;
            image.version = pending.version;
          end
          CONTEXT: begin
            vif.inputs[`FETCH_IN_REDIRECT_VALID_OFFSET+:`FETCH_IN_REDIRECT_VALID_WIDTH] = 1;
            vif.inputs[`FETCH_IN_REDIRECT_BITS_OFFSET+:`FETCH_IN_REDIRECT_BITS_WIDTH]=pending.target;
            privilege = pending.privilege;
            satp = pending.satp;
            sum = pending.sum;
            mxr = pending.mxr;
          end
          INVALIDATE: vif.inputs[`FETCH_IN_INVALIDATE_OFFSET+:`FETCH_IN_INVALIDATE_WIDTH] = 1;
          RESET_VECTOR: reset_vector = pending.target;
          BAD_FLUSH: vif.inputs[`FETCH_IN_FLUSH_OFFSET+:`FETCH_IN_FLUSH_WIDTH] = 1;
        endcase
      end
      vif.inputs[`FETCH_IN_RESETVECTOR_OFFSET+:`FETCH_IN_RESETVECTOR_WIDTH] = reset_vector;
      vif.inputs[`FETCH_IN_CONTEXT_PRIVILEGE_OFFSET+:`FETCH_IN_CONTEXT_PRIVILEGE_WIDTH] = privilege;
      vif.inputs[`FETCH_IN_CONTEXT_SATP_OFFSET+:`FETCH_IN_CONTEXT_SATP_WIDTH] = satp;
      vif.inputs[`FETCH_IN_CONTEXT_SUM_OFFSET+:`FETCH_IN_CONTEXT_SUM_WIDTH] = sum;
      vif.inputs[`FETCH_IN_CONTEXT_MXR_OFFSET+:`FETCH_IN_CONTEXT_MXR_WIDTH] = mxr;
      @(posedge vif.clock);
      if (pending != null) begin
        #1;
        seq_item_port.item_done();
        pending = null;
      end
    end
  endtask
endclass
