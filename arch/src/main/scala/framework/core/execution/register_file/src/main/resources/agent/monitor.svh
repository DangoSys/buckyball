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
        req.read0 = vif.read0;
        req.read1 = vif.read1;
        req.write = vif.write;
        req.address = vif.address;
        req.data = vif.data;
        rsp.data0 = vif.data0;
        rsp.data1 = vif.data1;
        requests.write(req);
        responses.write(rsp);
      end
    end
  endtask
endclass
