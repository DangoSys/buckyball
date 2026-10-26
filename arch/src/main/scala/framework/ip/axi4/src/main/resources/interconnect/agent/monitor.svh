class monitor extends uvm_monitor;
  `uvm_component_utils(monitor)
  virtual bus_if vif;
  uvm_analysis_port #(snapshot) ap;
  function new(string name, uvm_component parent);
    super.new(name, parent);
    ap = new("ap", this);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual bus_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "AXI monitor requires an interface")
  endfunction
  task run_phase(uvm_phase phase);
    snapshot observed;
    forever begin
      @(posedge vif.clock);
      observed = snapshot::type_id::create("observed");
      observed.ports = vif.ports;
      observed.reset = vif.reset;
      observed.outstanding = vif.outstanding;
      ap.write(observed);
    end
  endtask
endclass
