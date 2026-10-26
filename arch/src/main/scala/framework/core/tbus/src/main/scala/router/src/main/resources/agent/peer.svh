class peer extends uvm_component;
  `uvm_component_utils(peer)
  virtual tbus_if vif;
  fixture image;
  const int offsets[`TR_CLIENTS+`TR_EXTERNAL][15] = `TR_MESH_OFFSETS;
  bit [127:0] memory[`TR_BANKS][`TR_ROWS];
  bit [15:0] initialized[`TR_BANKS][`TR_ROWS];
  bit
      pending[`TR_CLIENTS+`TR_EXTERNAL],
      writing[`TR_CLIENTS+`TR_EXTERNAL],
      error[`TR_CLIENTS+`TR_EXTERNAL];
  bit [127:0] response[`TR_CLIENTS+`TR_EXTERNAL];
  int tag[`TR_CLIENTS+`TR_EXTERNAL], due[`TR_CLIENTS+`TR_EXTERNAL], cycle;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual tbus_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "TBus interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    forever begin
      @(posedge vif.clock);
      cycle++;
      for (int c = 0; c < `TR_CLIENTS + `TR_EXTERNAL; c++) begin
        if (vif.reset) pending[c] = 0;
        else begin
          if (pending[c] && vif.inputs[offsets[c][8]] && vif.outputs[offsets[c][9]]) pending[c] = 0;
          if (vif.outputs[offsets[c][0]] && vif.inputs[offsets[c][1]]) begin
            int bank_id = int'(vif.outputs[offsets[c][2]+:$clog2(`TR_BANKS)]);
            int row = int'(vif.outputs[offsets[c][3]+1+:16]);
            bit write_request = vif.outputs[offsets[c][3]];
            bit [15:0] mask = vif.outputs[offsets[c][5]+:16];
            if (pending[c] || bank_id >= `TR_BANKS || row >= `TR_ROWS)
              `uvm_fatal("BACKEND", "Invalid or overlapping Mesh request")
            pending[c] = 1;
            writing[c] = write_request;
            error[c] = image.error[c];
            tag[c] = int'(vif.outputs[offsets[c][6]+:`TR_TAG_BITS]);
            due[c] = cycle + 2 + c % 4;
            response[c] = 0;
            if (write_request && !error[c]) begin
              for (int b = 0; b < 16; b++)
              if (mask[b]) begin
                memory[bank_id][row][b*8+:8] = vif.outputs[offsets[c][4]+b*8+:8];
                initialized[bank_id][row][b] = 1;
              end
            end else if (!write_request) begin
              if (initialized[bank_id][row] != '1)
                `uvm_fatal("MEMORY", "Uninitialized Mesh bank read")
              response[c] = memory[bank_id][row];
            end
          end
        end
      end
      @(negedge vif.clock);
      vif.backend = '0;
      for (int c = 0; c < `TR_CLIENTS + `TR_EXTERNAL; c++) begin
        vif.backend[offsets[c][1]] = !pending[c] && cycle % 5 != c % 5;
        vif.backend[offsets[c][8]] = pending[c] && cycle >= due[c];
        vif.backend[offsets[c][10]+:128] = response[c];
        vif.backend[offsets[c][11]+:16] = '1;
        vif.backend[offsets[c][12]+:`TR_TAG_BITS] = tag[c];
        vif.backend[offsets[c][13]+:2] = {writing[c], error[c]};
        vif.backend[offsets[c][14]] = 1;
      end
    end
  endtask
endclass
