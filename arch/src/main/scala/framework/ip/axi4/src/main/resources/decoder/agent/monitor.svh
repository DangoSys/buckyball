class monitor extends uvm_monitor;
  `uvm_component_utils(monitor)
  virtual decoder_if vif;
  uvm_analysis_port #(cycle_item) ap;
  function new(string name, uvm_component parent);
    super.new(name, parent);
    ap = new("ap", this);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual decoder_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "Decoder interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    forever begin
      cycle_item t;
      @(posedge vif.clock);
      t = cycle_item::type_id::create("observed");
      t.ports = vif.ports;
      t.reset = vif.reset;
      ap.write(t);
    end
  endtask
endclass
