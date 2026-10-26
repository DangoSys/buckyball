class monitor extends uvm_monitor;
  `uvm_component_utils(monitor)
  virtual link_if vif;
  uvm_analysis_port #(cycle_item) ap;
  function new(string name, uvm_component parent);
    super.new(name, parent);
    ap = new("ap", this);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual link_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "CHI link monitor requires an interface")
  endfunction
  task run_phase(uvm_phase phase);
    cycle_item t;
    forever begin
      @(posedge vif.clock);
      t = cycle_item::type_id::create("observed");
      t.reset = vif.reset;
      t.sampled = vif.pins;
      ap.write(t);
    end
  endtask
endclass
