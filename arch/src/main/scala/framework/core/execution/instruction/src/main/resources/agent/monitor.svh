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
        req.bits = vif.bits;
        rsp.expanded = vif.illegal ? 0 : vif.expanded;
        rsp.compressed = vif.compressed;
        rsp.illegal = vif.illegal;
        rsp.opcode = vif.illegal ? 0 : vif.opcode;
        rsp.funct3 = vif.illegal ? 0 : vif.funct3;
        rsp.funct7 = vif.illegal ? 0 : vif.funct7;
        rsp.rs1 = vif.illegal ? 0 : vif.rs1;
        rsp.rs2 = vif.illegal ? 0 : vif.rs2;
        rsp.rd = vif.illegal ? 0 : vif.rd;
        rsp.csr_address = vif.illegal ? 0 : vif.csr_address;
        rsp.immediate = vif.illegal ? 0 : vif.immediate;
        rsp.atomic = vif.illegal ? 0 : vif.atomic;
        rsp.function_code = vif.illegal ? 0 : vif.function_code;
        rsp.word = vif.illegal ? 0 : vif.word;
        rsp.immediate_mode = vif.illegal ? 0 : vif.immediate_mode;
        requests.write(req);
        responses.write(rsp);
      end
    end
  endtask
endclass
