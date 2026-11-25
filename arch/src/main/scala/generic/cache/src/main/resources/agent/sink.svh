class sink extends uvm_component;
  `uvm_component_utils(sink)
  virtual cache_response_if vif;
  bit blocked = 0;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual cache_response_if)::get(this, "", "response", vif))
      `uvm_fatal("VIF", "Cache response interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    forever begin
      @(vif.cb);
      vif.cb.ready <= !vif.reset && !blocked;
    end
  endtask
endclass
