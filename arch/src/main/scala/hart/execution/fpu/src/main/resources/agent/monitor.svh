class monitor extends uvm_monitor;
  `uvm_component_utils(monitor)
  virtual port_if vif;
  uvm_analysis_port #(request_item) requests;
  uvm_analysis_port #(response_item) responses;
  uvm_analysis_port #(bit) reset_ap;
  function new(string name, uvm_component parent);
    super.new(name, parent);
    requests  = new("requests", this);
    responses = new("responses", this);
    reset_ap  = new("reset_ap", this);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual port_if)::get(this, "", "vif", vif))
      `uvm_fatal("INTERFACE", "FPU monitor requires ports")
  endfunction
  task run_phase(uvm_phase phase);
    bit pending = 0;
    request_item req;
    response_item rsp;
    forever begin
      @(posedge vif.clock);
      if (vif.reset) begin
        pending = 0;
        reset_ap.write(1);
      end else begin
        if (vif.request_valid && vif.request_ready) begin
          if (pending) `uvm_fatal("OUTSTANDING", "FPU accepted a second request")
          req = request_item::type_id::create("request");
          req.instruction = vif.instruction;
          req.integer_value = vif.integer_value;
          req.a = vif.a;
          req.rhs = vif.rhs;
          req.c = vif.c;
          req.rounding = vif.rounding;
          req.stall = vif.stall;
          req.cancel = vif.cancel;
          requests.write(req);
          pending = 1;
        end else if (pending && vif.request_ready)
          `uvm_fatal("BUSY", "FPU reopened its request port before completion")
        if (vif.response_valid && !pending)
          `uvm_fatal("EXTRA_RESULT", "FPU produced an unsolicited result")
        if (vif.response_valid && vif.response_ready) begin
          rsp = response_item::type_id::create("response");
          rsp.value = vif.illegal ? 0 : vif.value;
          rsp.flags = vif.flags;
          rsp.integer_result = vif.illegal ? 0 : vif.integer_result;
          rsp.illegal = vif.illegal;
          responses.write(rsp);
          pending = 0;
        end
      end
    end
  endtask
endclass
