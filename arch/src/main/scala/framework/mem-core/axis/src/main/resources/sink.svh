class sink extends uvm_component;
  `uvm_component_utils(sink)

  virtual axis_if vif;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual axis_if)::get(this, "", "vif", vif)) begin
      `uvm_fatal("AXIS_VIF", "sink requires vif")
    end
  endfunction

  task run_phase(uvm_phase phase);
    int unsigned cycle = 0;
    vif.tready <= 1'b0;
    forever begin
      @(negedge vif.clock);
      cycle++;
      vif.tready <= !vif.reset && cycle[1:0] == 2'b11;
    end
  endtask
endclass
