class transport_item extends uvm_sequence_item;
  `uvm_object_utils_begin(transport_item)
    `uvm_field_int(value, UVM_ALL_ON)
  `uvm_object_utils_end
  logic [`COH_DAT_WIDTH-1:0] value;
  function new(string name = "flit");
    super.new(name);
  endfunction
endclass
class transport_monitor extends uvm_component;
  `uvm_component_utils(transport_monitor)
  virtual home_transport_if vif;
  result_board #(transport_item, 0) boards[6][`COH_AGENTS];
  covergroup behavior with function sample (int channel, int node);
    option.per_instance = 1;
    channels: coverpoint channel {bins all[] = {[0 : 5]};}
    nodes: coverpoint node {bins all[] = {[1 : `COH_AGENTS]};}
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    behavior = new();
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual home_transport_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "Home transport interface is required")
    foreach (boards[c, n])
    boards[c][n] = result_board#(transport_item, 0)::type_id::create(
        $sformatf("channel%0d_node%0d", c, n + 1), this);
  endfunction
  task run_phase(uvm_phase phase);
    forever begin
      @(posedge vif.clock);
      foreach (boards[c, n]) if (vif.reset) boards[c][n].cancel_pending();
      if (!vif.reset)
        for (int c = 0; c < 6; c++) begin
          if (vif.expected_valid[c]) begin
            transport_item t = transport_item::type_id::create("expected");
            int node = vif.expected_node[c];
            if (node < 1 || node > `COH_AGENTS)
              `uvm_fatal("NODE", "Unexpected source or target node at Home")
            t.value = vif.expected_bits[c];
            boards[c][node-1].write_expected(t);
            behavior.sample(c, node);
          end
          if (vif.actual_valid[c]) begin
            transport_item t = transport_item::type_id::create("actual");
            int node = vif.actual_node[c];
            if (node < 1 || node > `COH_AGENTS)
              `uvm_fatal("NODE", "Unexpected delivered node at Home")
            t.value = vif.actual_bits[c];
            boards[c][node-1].write_actual(t);
          end
        end
    end
  endtask
endclass
