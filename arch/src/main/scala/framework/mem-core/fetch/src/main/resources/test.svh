class protocol_test extends ip_test;
  `uvm_component_utils(protocol_test)
  virtual fetch_control_if control;
  virtual stream_if #(`FETCH_REQUEST_WIDTH) request;
  virtual stream_if #(`FETCH_RESPONSE_WIDTH) response;
  virtual stream_if #(`FETCH_PACKET_WIDTH) packet;
  virtual stream_if #(1) maintenance, maintained;
  int checked = 0, instructions = 0, reads = 0, drains = 0, barriers = 0, cancellations = 0;
  longint unsigned cursor;
  bit pending_half = 0;
  bit [15:0] first_half;
  longint unsigned partial_pc;
  bit [63:0] saved_word;
  bit [`FETCH_REQUEST_WIDTH-1:0] saved_request;
  int version = 0;
  function new(string name, uvm_component parent);
    super.new(name, parent);
    timeout = 200us;
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual fetch_control_if)::get(
            this, "", "control", control
        ) || !uvm_config_db#(virtual stream_if #(`FETCH_REQUEST_WIDTH))::get(
            this, "", "request", request
        ) || !uvm_config_db#(virtual stream_if #(`FETCH_RESPONSE_WIDTH))::get(
            this, "", "response", response
        ) || !uvm_config_db#(virtual stream_if #(`FETCH_PACKET_WIDTH))::get(
            this, "", "packet", packet
        ) || !uvm_config_db#(virtual stream_if #(1))::get(
            this, "", "maintenance", maintenance
        ) || !uvm_config_db#(virtual stream_if #(1))::get(
            this, "", "maintained", maintained
        ))
      `uvm_fatal("VIF", "Fetch interfaces missing")
  endfunction
  task redirect(longint unsigned target, bit flush = 0);
    control.cb.redirect_pc <= target;
    control.cb.redirect_valid <= 1;
    control.cb.flush <= flush;
    @(control.cb);
    control.cb.redirect_valid <= 0;
    control.cb.flush <= 0;
    cursor = target;
    pending_half = 0;
  endtask
  task offered(longint unsigned expected_pc);
    do @(request.sample); while (!request.sample.valid);
    saved_request = request.sample.bits;
    if (saved_request[
        `FF(REQUEST, ADDR)
        ] !== (expected_pc & ~64'h7) || saved_request[
        `FF(REQUEST, CONTEXT_PRIVILEGE)
        ] !== control.privilege || saved_request[
        `FF(REQUEST, CONTEXT_SATP)
        ] !== control.satp || saved_request[
        `FF(REQUEST, CONTEXT_SUM)
        ] !== control.sum || saved_request[
        `FF(REQUEST, CONTEXT_MXR)
        ] !== control.mxr || saved_request[
        `FF(REQUEST, EXECUTE)
        ] !== 1)
      `uvm_fatal("READ", "Word request changed PC/context/execute metadata")
    saved_word = fetch_ref_word(expected_pc & ~64'h7, version);
  endtask
  task accept_word();
    request.ready <= 1;
    @(request.sample);
    request.ready <= 0;
    reads++;
  endtask
  task reply(bit pf = 0, bit af = 0);
    response.bits <= '0;
    response.bits[`FF(RESPONSE, DATA)] <= saved_word;
    response.bits[`FF(RESPONSE, PAGEFAULT)] <= pf;
    response.bits[`FF(RESPONSE, ACCESSFAULT)] <= af;
    response.valid <= 1;
    do @(response.sample); while (!response.sample.ready);
    response.valid <= 0;
  endtask
  task check_packet(bit pf = 0, bit af = 0);
    bit [31:0] data;
    bit [ 1:0] mask;
    bit [63:0] pc;
    do @(packet.sample); while (!packet.sample.valid);
    pc   = packet.sample.bits[`FF(PACKET, PC)];
    data = packet.sample.bits[`FF(PACKET, DATA)];
    mask = packet.sample.bits[`FF(PACKET, MASK)];
    if (pc !== cursor || data !== (pf || af ? 0 : fetch_ref_group(
            cursor, version
        )) || mask !== (cursor[1] ? 2'b10 : 2'b11) || packet.sample.bits[
        `FF(PACKET, PAGEFAULT)
        ] !== pf || packet.sample.bits[
        `FF(PACKET, ACCESSFAULT)
        ] !== af)
      `uvm_fatal("PACKET", $sformatf(
                 "Fetch packet mismatch cursor=%h got=%h data=%h", cursor, pc, data))
    // Independent instruction reconstruction exercises the same packet boundaries that IBuf consumes.
    for (int lane = 0; lane < 2; lane++)
      if (mask[lane]) begin
        longint unsigned here = (cursor & ~64'h3) + lane * 2;
        bit [15:0] halfword = data[lane*16+:16];
        if (pf || af) begin
          if (pending_half && here !== partial_pc + 2)
            `uvm_fatal("SECOND_HALF", "Fault did not apply to the second halfword")
          pending_half = 0;
        end else if (pending_half) begin
          if ({halfword, first_half} !== fetch_ref_instruction(
                  partial_pc, version
              ) || here !== partial_pc + 2)
            `uvm_fatal("RVC",
                       "32-bit instruction split across groups was reconstructed incorrectly")
          pending_half = 0;
          instructions++;
        end else if (halfword[1:0] != 2'b11) begin
          if (halfword !== fetch_ref_instruction(here, version))
            `uvm_fatal("RVC", "Compressed instruction mismatch")
          instructions++;
        end else begin
          first_half   = halfword;
          partial_pc   = here;
          pending_half = 1;
        end
      end
    repeat (3) @(control.cb);
    packet.ready <= 1;
    @(packet.sample);
    packet.ready <= 0;
    checked++;
    cursor = (cursor & ~64'h3) + 4;
  endtask
  task fetch_one(bit pf = 0, bit af = 0);
    offered(cursor);
    repeat (3) @(control.cb);
    accept_word();
    repeat (2) @(control.cb);
    reply(pf, af);
    check_packet(pf, af);
  endtask
  task discarded_reply();
    reply();
    @(control.cb);
    if (packet.valid) `uvm_fatal("CANCEL", "Old fetch response escaped redirection")
    drains++;
  endtask
  task fence_ack();
    do @(maintenance.sample); while (!maintenance.sample.valid);
    repeat (3) begin
      @(control.cb);
      if (request.valid || packet.valid)
        `uvm_fatal("FENCE", "Fetch proceeded before maintenance request")
    end
    maintenance.ready <= 1;
    @(maintenance.sample);
    maintenance.ready <= 0;
    repeat (3) begin
      @(control.cb);
      if (request.valid || packet.valid)
        `uvm_fatal("FENCE", "Fetch proceeded before maintenance completed")
    end
`ifdef FETCH_BAD_ACK
    maintained.bits <= 0;
`else
    maintained.bits <= 1;
`endif
    maintained.valid <= 1;
    do @(maintained.sample); while (!maintained.sample.ready);
    maintained.valid <= 0;
    maintained.bits  <= 0;
    barriers++;
  endtask
  task reset_fetch(int phase_index);
    offered(cursor);
    if (phase_index != 0) begin
      accept_word();
      if (phase_index == 2) begin
        reply();
        do @(packet.sample); while (!packet.sample.valid);
      end
    end
    control.cb.reset <= 1;
    response.valid <= 0;
    maintained.valid <= 1;
    maintained.bits <= 0;
    control.cb.redirect_valid <= 0;
    control.cb.flush <= 0;
    control.cb.privilege <= 2;
    control.cb.satp <= '1;
    control.reset_vector <= '1;
    repeat (2) @(control.cb);
    control.cb.redirect_valid <= 1;
    control.cb.redirect_pc <= '1;
    control.cb.flush <= 1;
    repeat (2) @(control.cb);
    control.cb.redirect_valid <= 0;
    control.cb.flush <= 1;
    control.reset_vector <= 0;
    repeat (2) @(control.cb);
    control.cb.flush <= 0;
    maintained.valid <= 0;
    control.cb.privilege <= 3;
    control.cb.satp <= 0;
    repeat (2) @(control.cb);
    control.cb.reset <= 0;
    cursor = control.reset_vector;
    pending_half = 0;
    cancellations++;
    fetch_one();
  endtask
  task execute();
    control.reset = 1;
`ifdef FETCH_BAD_RESET
    control.reset_vector = 1;
`else
    control.reset_vector = 0;
`endif
    control.redirect_valid = 0;
    control.redirect_pc = 0;
    control.flush = 0;
`ifdef FETCH_BAD_PRIVILEGE
    control.privilege = 2;
`else
    control.privilege = 3;
`endif
    control.satp = 0;
    control.sum = 0;
    control.mxr = 0;
    request.ready = 0;
    response.valid = 0;
    response.bits = '0;
    packet.ready = 0;
    maintenance.ready = 0;
    maintained.valid = 0;
    maintained.bits = 1;
    cursor = 0;
    repeat (4) @(control.cb);
    control.cb.reset <= 0;
`ifdef FETCH_BAD_REDIRECT
    redirect(1);
    repeat (10) @(control.cb);
    `uvm_fatal("NO_ASSERT", "Odd instruction PC was accepted")
`elsif FETCH_BAD_FLUSH
    control.cb.flush <= 1;
    repeat (10) @(control.cb);
    `uvm_fatal("NO_ASSERT", "Flush without restart was accepted")
`else
    for (int i = 0; i < 8; i++) fetch_one();
    // Zero-latency word owner: response VALID begins in the request handshake cycle.
    offered(cursor);
    fork
      accept_word();
      reply();
    join
    check_packet();
    // A stalled offer is immutable even after redirect and live context changes.
    offered(cursor);
    control.cb.privilege <= 2;
    control.reset_vector <= 1;
    @(control.cb);
    control.cb.privilege <= 0;
    control.reset_vector <= 0;
    control.cb.satp <= 64'h8000123456789000;
    control.cb.context_sum <= 1;
    control.cb.mxr <= 1;
    redirect(2);
    repeat (3) @(control.cb);
    if (request.bits !== saved_request || !request.valid)
      `uvm_fatal("OFFER", "Redirect withdrew or changed a backpressured word offer")
    accept_word();
    redirect(6);
    discarded_reply();
    for (int i = 0; i < 4; i++) fetch_one();
    // Same-cycle response and redirect must drain the old word and suppress the packet.
    offered(cursor);
    accept_word();
    fork
      redirect(62);
      begin
        discarded_reply();
      end
    join
    for (int i = 0; i < 4; i++) fetch_one();
    // A redirect in the same cycle as accepting the old word still drains that word.
    offered(cursor);
    fork
      accept_word();
      redirect(66);
    join
    discarded_reply();
    for (int i = 0; i < 3; i++) fetch_one();
    // Cancel an already buffered packet under CPU backpressure.
    offered(cursor);
    accept_word();
    reply();
    do @(packet.sample); while (!packet.sample.valid);
    packet.ready <= 1;
    redirect(126);
    packet.ready <= 0;
    for (int i = 0; i < 3; i++) fetch_one();
    // At PC=4094 a 32-bit instruction starts at the last halfword of the page.
    redirect(4094);
    fetch_one();
    if (!pending_half)
      `uvm_fatal("SECOND_HALF", "Page boundary did not leave a partial instruction")
    fetch_one(1, 0);
    redirect(8192);
    fetch_one(0, 1);
    // Flush must drain an old accepted read before requesting external maintenance.
    offered(cursor);
    accept_word();
    version = 1;
    redirect(0, 1);
    discarded_reply();
    fence_ack();
    for (int i = 0; i < 4; i++) fetch_one();
    // A fence arriving with a maintenance handshake needs another ordered acknowledgement.
    offered(cursor);
    accept_word();
    reply();
    do @(packet.sample); while (!packet.sample.valid);
    redirect(18, 1);
    do @(maintenance.sample); while (!maintenance.sample.valid);
    fork
      begin
        maintenance.ready <= 1;
        @(maintenance.sample);
        maintenance.ready <= 0;
      end
      redirect(30, 1);
    join
    repeat (3) @(control.cb);
    maintained.valid <= 1;
    maintained.bits  <= 1;
    do @(maintained.sample); while (!maintained.sample.ready);
    maintained.valid <= 0;
    maintained.bits  <= 0;
    barriers++;
    fence_ack();
    for (int i = 0; i < 3; i++) fetch_one();
    // Byte-memory patterns vary all transported instruction bits without duplicating the execution decoder.
    for (int image = 2; image < 36; image++) begin
      version = image;
      control.cb.privilege <= image % 3 == 0 ? 3 : (image % 3 == 1 ? 1 : 0);
      control.cb.satp <= image[0] ? 64'h8fffffffffffffff : 0;
      control.cb.context_sum <= image[0];
      control.cb.mxr <= !image[0];
      redirect(image * 64);
      for (int i = 0; i < 3; i++) fetch_one();
    end
    control.cb.privilege <= 3;
    control.cb.satp <= 0;
    redirect(64'h2000);
    fetch_one();
    for (int bit_index = 3; bit_index < 64; bit_index++) begin
      redirect(64'h1 << bit_index);
      fetch_one(0, bit_index >= 44);
    end
    for (int phase_index = 0; phase_index < 3; phase_index++) reset_fetch(phase_index);
    // Joint reset cancels an accepted maintenance transaction and ignores its reset-cycle payload.
    offered(cursor);
    accept_word();
    reply();
    do @(packet.sample); while (!packet.sample.valid);
    redirect(0, 1);
    do @(maintenance.sample); while (!maintenance.sample.valid);
    maintenance.ready <= 1;
    @(maintenance.sample);
    maintenance.ready <= 0;
    control.cb.reset  <= 1;
    maintained.valid  <= 1;
    maintained.bits   <= 0;
    repeat (2) @(control.cb);
    maintained.valid <= 0;
    control.cb.reset <= 0;
    cursor = control.reset_vector;
    pending_half = 0;
    cancellations++;
    fetch_one();
    // READY can be high with VALID=0; unrelated payload cannot create a transaction.
    request.ready <= 1;
    maintenance.ready <= 1;
    control.cb.redirect_pc <= '1;
    @(control.cb);
    control.cb.redirect_pc <= 0;
    // Explicitly terminate the autonomous producer and cancel any trailing halfword by joint reset.
    control.cb.reset <= 1;
    response.valid <= 0;
    maintained.valid <= 0;
    pending_half = 0;
    repeat (2) @(control.cb);
    if (request.valid || packet.valid || maintenance.valid)
      `uvm_fatal("RESET", "Reset retained a live frontend transaction")
    `uvm_info("FETCH", $sformatf(
              "Checked %0d packets, reconstructed %0d instructions, %0d accepted reads, %0d stale drains, %0d maintenance acknowledgements, %0d joint-reset cancellations",
              checked,
              instructions,
              reads,
              drains,
              barriers,
              cancellations
              ), UVM_LOW)
`endif
  endtask
endclass
