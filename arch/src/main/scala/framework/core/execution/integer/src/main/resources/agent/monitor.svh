class monitor extends uvm_monitor;
  `uvm_component_utils(monitor)
  virtual port_if vif;
  uvm_analysis_port #(request_item) requests;
  uvm_analysis_port #(response_item) responses;
  function new(string name, uvm_component parent);
    super.new(name, parent);
    requests  = new("requests", this);
    responses = new("responses", this);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual port_if)::get(this, "", "vif", vif))
      `uvm_fatal("INTERFACE", "Monitor interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    request_item  req;
    response_item rsp;
    forever begin
      @(posedge vif.clock);
      if (!vif.reset && vif.sample) begin
        req = request_item::type_id::create("request");
        rsp = response_item::type_id::create("response");
        req.a = vif.a;
        req.rhs = vif.rhs;
        req.fn = vif.fn;
        req.dw = vif.dw;
        rsp.value = vif.value;
        rsp.adder = vif.adder;
        rsp.comparison = vif.comparison;
        requests.write(req);
        responses.write(rsp);
      end
    end
  endtask
endclass
