class peer extends uvm_component;
  `uvm_component_utils(peer)
  virtual router_if vif;
  const int banks[`BR_BANKS][17] = `BR_BANK_OFFSETS;
  logic [`BR_DATA_BITS-1:0] memory[`BR_BANKS][`BR_ROWS];
  bit initialized[`BR_BANKS][`BR_ROWS];
  int cycle;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual router_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "Router interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    forever begin
      logic [`BR_INPUT_BITS-1:0] next_backend;
      @(posedge vif.clock);
      next_backend = '0;
      cycle++;
      if (!vif.reset)
        for (int b = 0; b < `BR_BANKS; b++) begin
          if (vif.outputs[banks[b][14]])
            for (int row = 0; row < `BR_ROWS; row++) begin
              memory[b][row] = '0;
              initialized[b][row] = 1;
            end
          if (vif.outputs[banks[b][0]] && vif.inputs[banks[b][1]]) begin
            int row = int'(vif.outputs[banks[b][2]+:16]);
            if (row >= `BR_ROWS || !initialized[b][row])
              `uvm_fatal("MEMORY", "Uninitialized physical bank read")
            next_backend[banks[b][3]] = 1;
            next_backend[banks[b][5]+:`BR_DATA_BITS] = memory[b][row];
          end
          if (vif.outputs[banks[b][6]] && vif.inputs[banks[b][7]]) begin
            int row = int'(vif.outputs[banks[b][8]+:16]);
            if (row >= `BR_ROWS) `uvm_fatal("MEMORY", "Physical bank address overflow")
            for (int byte_index = 0; byte_index < `BR_MASK_BITS; byte_index++)
            if (vif.outputs[banks[b][10]+byte_index])
              memory[b][row][byte_index*8+:8] = vif.outputs[banks[b][9]+byte_index*8+:8];
            // Every row is explicitly initialized by a full write before partial writes.
            initialized[b][row] = 1;
            next_backend[banks[b][11]] = 1;
            next_backend[banks[b][13]] = 1;
          end
        end
      @(negedge vif.clock);
      for (int b = 0; b < `BR_BANKS; b++) begin
        next_backend[banks[b][1]] = cycle % 4 != b % 4;
        next_backend[banks[b][7]] = cycle % 5 != b % 5;
      end
      vif.backend = next_backend;
    end
  endtask
endclass
