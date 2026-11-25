class peer_driver extends uvm_component;
  `uvm_component_utils(peer_driver)
  virtual fetch_if vif;
  model reference;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual fetch_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "Fetch backend interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    forever begin
      @(negedge vif.clock);
      #1;
      vif.inputs[`FETCH_IN_PACKET_READY_OFFSET+:`FETCH_IN_PACKET_READY_WIDTH]=!vif.reset&&reference.packet_ready&&reference.cycle%5>=2;
      vif.inputs[`FETCH_IN_REQUEST_READY_OFFSET+:`FETCH_IN_REQUEST_READY_WIDTH]=!vif.reset&&!reference.hold_request&&reference.cycle%4!=0;
      vif.inputs[`FETCH_IN_RESPONSE_VALID_OFFSET+:`FETCH_IN_RESPONSE_VALID_WIDTH]=!vif.reset&&reference.word!=null&&!reference.hold_word&&reference.cycle>=reference.word.due;
      if (reference.word != null) begin
        vif.inputs[`FETCH_IN_RESPONSE_BITS_DATA_OFFSET+:`FETCH_IN_RESPONSE_BITS_DATA_WIDTH]=reference.word.data;
        vif.inputs[`FETCH_IN_RESPONSE_BITS_LINE_OFFSET+:`FETCH_IN_RESPONSE_BITS_LINE_WIDTH]=reference.word.line;
        vif.inputs[`FETCH_IN_RESPONSE_BITS_LINEEXECUTABLE_OFFSET+:`FETCH_IN_RESPONSE_BITS_LINEEXECUTABLE_WIDTH]=reference.word.executable;
        vif.inputs[`FETCH_IN_RESPONSE_BITS_PAGEFAULT_OFFSET+:`FETCH_IN_RESPONSE_BITS_PAGEFAULT_WIDTH]=reference.word.pf;
        vif.inputs[`FETCH_IN_RESPONSE_BITS_ACCESSFAULT_OFFSET+:`FETCH_IN_RESPONSE_BITS_ACCESSFAULT_WIDTH]=reference.word.af;
      end
      vif.inputs[`FETCH_IN_MAINTENANCE_READY_OFFSET+:`FETCH_IN_MAINTENANCE_READY_WIDTH]=!vif.reset&&!reference.hold_maintenance;
      vif.inputs[`FETCH_IN_MAINTAINED_VALID_OFFSET+:`FETCH_IN_MAINTAINED_VALID_WIDTH]=!vif.reset&&reference.maintenance_active&&!reference.hold_ack&&reference.cycle>=reference.ack_due;
      vif.inputs[`FETCH_IN_MAINTAINED_BITS_OFFSET+:`FETCH_IN_MAINTAINED_BITS_WIDTH] = 1;
    end
  endtask
endclass
class invalid_peer extends uvm_component;
  `uvm_component_utils(invalid_peer)
  virtual fetch_if vif;
  program_image image;
  bit pending_word = 0, pending_ack = 0;
  longint unsigned address;
  string scenario;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual fetch_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "Fetch assertion backend is required")
    if (!$value$plusargs("CASE=%s", scenario))
      `uvm_fatal("CASE", "Fetch assertion scenario is required")
  endfunction
  task run_phase(uvm_phase phase);
    logic [511:0] line;
    forever begin
      @(posedge vif.clock);
      if (vif.reset) begin
        pending_word = 0;
        pending_ack  = 0;
      end else begin
        if (
            `FO(vif.outputs, REQUEST_VALID)
            && vif.inputs[`FETCH_IN_REQUEST_READY_OFFSET+:`FETCH_IN_REQUEST_READY_WIDTH]) begin
          pending_word = 1;
          address = `FO(vif.outputs, REQUEST_BITS_ADDR);
        end
        if (vif.inputs[`FETCH_IN_RESPONSE_VALID_OFFSET+:`FETCH_IN_RESPONSE_VALID_WIDTH] &&
            `FO(vif.outputs, RESPONSE_READY)
            )
          pending_word = 0;
        if (
            `FO(vif.outputs, MAINTENANCE_VALID)
            && vif.inputs[`FETCH_IN_MAINTENANCE_READY_OFFSET+:`FETCH_IN_MAINTENANCE_READY_WIDTH])
          pending_ack = 1;
        if (vif.inputs[`FETCH_IN_MAINTAINED_VALID_OFFSET+:`FETCH_IN_MAINTAINED_VALID_WIDTH] &&
            `FO(vif.outputs, MAINTAINED_READY)
            )
          pending_ack = 0;
      end
      @(negedge vif.clock);
      #1;
      vif.inputs[`FETCH_IN_PACKET_READY_OFFSET+:`FETCH_IN_PACKET_READY_WIDTH] = 0;
      vif.inputs[`FETCH_IN_REQUEST_READY_OFFSET+:`FETCH_IN_REQUEST_READY_WIDTH] = !vif.reset;
      vif.inputs[`FETCH_IN_RESPONSE_VALID_OFFSET+:`FETCH_IN_RESPONSE_VALID_WIDTH]=pending_word&&!vif.reset;
      vif.inputs[`FETCH_IN_RESPONSE_BITS_DATA_OFFSET+:`FETCH_IN_RESPONSE_BITS_DATA_WIDTH]=fetch_ref_word(
          address, image.version);
      for (int word = 0; word < 8; word++)
      line[word*64+:64] = fetch_ref_word((address & ~64'h3f) + word * 8, image.version);
      vif.inputs[`FETCH_IN_RESPONSE_BITS_LINE_OFFSET+:`FETCH_IN_RESPONSE_BITS_LINE_WIDTH] = line;
      vif.inputs[`FETCH_IN_RESPONSE_BITS_LINEEXECUTABLE_OFFSET+:`FETCH_IN_RESPONSE_BITS_LINEEXECUTABLE_WIDTH]=0;
      vif.inputs[`FETCH_IN_RESPONSE_BITS_PAGEFAULT_OFFSET+:`FETCH_IN_RESPONSE_BITS_PAGEFAULT_WIDTH]=0;
      vif.inputs[`FETCH_IN_RESPONSE_BITS_ACCESSFAULT_OFFSET+:`FETCH_IN_RESPONSE_BITS_ACCESSFAULT_WIDTH]=0;
      vif.inputs[`FETCH_IN_MAINTENANCE_READY_OFFSET+:`FETCH_IN_MAINTENANCE_READY_WIDTH]=!vif.reset;
      vif.inputs[`FETCH_IN_MAINTAINED_VALID_OFFSET+:`FETCH_IN_MAINTAINED_VALID_WIDTH]=pending_ack&&!vif.reset;
      vif.inputs[`FETCH_IN_MAINTAINED_BITS_OFFSET+:`FETCH_IN_MAINTAINED_BITS_WIDTH]=scenario!="ack";
    end
  endtask
endclass
