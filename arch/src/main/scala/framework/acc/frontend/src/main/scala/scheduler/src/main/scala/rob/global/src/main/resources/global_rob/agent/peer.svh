class peer_driver extends uvm_component;
  `uvm_component_utils(peer_driver)
  virtual global_if vif;
  model reference;
  int query_offsets[`GROB_ENTRIES][9] = `GROB_QUERY_OFFSETS;
  int hazard_offsets[`GROB_ENTRIES] = `GROB_HAZARD_OFFSETS;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual global_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "Global ROB peer interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    bank_access_t renamed;
    bit ready, rename, reset_read;
    bit [1:0] stale_valid;
    int stale0, stale1, chosen;
    forever begin
      @(negedge vif.clock);
      #1;
      reference.predict_bat(vif.inputs, renamed, ready, rename, reset_read, stale_valid, stale0,
                            stale1);
      vif.inputs[`GROB_IN_BAT_READY_OFFSET+:`GROB_IN_BAT_READY_WIDTH] = ready;
      vif.inputs[`GROB_IN_RENAMED_OFFSET+:`GROB_IN_RENAMED_BITS] = renamed;
      vif.inputs[`GROB_IN_BAT_STALE_0_VALID_OFFSET+:`GROB_IN_BAT_STALE_0_VALID_WIDTH]=stale_valid[0];
      vif.inputs[`GROB_IN_BAT_STALE_0_BITS_OFFSET+:`GROB_IN_BAT_STALE_0_BITS_WIDTH] = stale0;
      vif.inputs[`GROB_IN_BAT_STALE_1_VALID_OFFSET+:`GROB_IN_BAT_STALE_1_VALID_WIDTH]=stale_valid[1];
      vif.inputs[`GROB_IN_BAT_STALE_1_BITS_OFFSET+:`GROB_IN_BAT_STALE_1_BITS_WIDTH] = stale1;
      vif.inputs[`GROB_IN_ISSUE_READY_OFFSET+:`GROB_IN_ISSUE_READY_WIDTH]=!vif.reset&&!reference.hold_issue&&reference.cycle%5!=0;
      vif.inputs[`GROB_IN_MAPPING_READY_OFFSET+:`GROB_IN_MAPPING_READY_WIDTH]=!vif.reset&&!reference.hold_mapping&&reference.cycle%4!=0;
      for (int lane = 0; lane < `GROB_ENTRIES; lane++) begin
        bit hazard = 0;
        for (int port = 0; port < 3; port++) begin
          if (vif.outputs[query_offsets[lane][port*2]])
            hazard |= reference.busy[vif.outputs[query_offsets[lane][port*2+1]+:`GROB_BANK_BITS]];
        end
        vif.inputs[hazard_offsets[lane]] = hazard;
      end
      if (vif.reset) begin
        vif.inputs[`GROB_IN_COMPLETE_VALID_OFFSET+:`GROB_IN_COMPLETE_VALID_WIDTH] = 0;
        vif.inputs[`GROB_IN_COMPLETE_BITS_OFFSET+:`GROB_IN_COMPLETE_BITS_WIDTH]   = 0;
      end else begin
        if (reference.active_completion == null && !reference.hold_complete) begin
          chosen = -1;
          foreach (reference.completion_queue[i]) begin
            if(reference.completion_queue[i].due<=reference.cycle&&(chosen<0||reference.reverse_complete))
              chosen = i;
          end
          if (chosen >= 0) begin
            reference.active_completion = reference.completion_queue[chosen];
            reference.completion_queue.delete(chosen);
          end
        end
        vif.inputs[`GROB_IN_COMPLETE_VALID_OFFSET+:`GROB_IN_COMPLETE_VALID_WIDTH]=reference.active_completion!=null;
        vif.inputs[`GROB_IN_COMPLETE_BITS_OFFSET+:`GROB_IN_COMPLETE_BITS_WIDTH]=reference.active_completion!=null?reference.active_completion.slot:0;
      end
    end
  endtask
endclass
