class monitor extends uvm_monitor;
  `uvm_component_utils(monitor)
  virtual fetch_if vif;
  uvm_analysis_port #(sample_item) ap;
  function new(string name, uvm_component parent);
    super.new(name, parent);
    ap = new("ap", this);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual fetch_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "Fetch monitor interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    sample_item t;
    forever begin
      @(posedge vif.clock);
      t = sample_item::type_id::create("observed");
      t.inputs = vif.inputs;
      t.outputs = vif.outputs;
      t.reset = vif.reset;
      ap.write(t);
    end
  endtask
endclass
