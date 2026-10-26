class model extends uvm_subscriber #(sample_item);
  `uvm_component_utils(model)
  program_image image;
  sample_item latest;
  logic [`FETCH_INPUT_BITS-1:0] previous_inputs = '0;
  longint unsigned cursor = 0, last_pc = 0;
  bit last_pf = 0, last_af = 0, last_valid = 0, offered_packet = 0;
  int epoch = 0, cycle = 0, checked = 0, reads = 0, replies = 0, reset_words = 0, drained_words = 0;
  int maintenance_requests = 0, maintenance_acks = 0, reset_maintenance = 0, ack_due = 0;
  bit needs_maintenance = 0, maintenance_active = 0;
  word_transaction offer, word;
  bit
      packet_ready = 0,
      hold_request = 0,
      hold_word = 0,
      hold_maintenance = 0,
      hold_ack = 0,
      line_executable = 1;
  bit pf_enabled = 0, af_enabled = 0;
  longint unsigned pf_address = 0, af_address = 0;
  uvm_analysis_port #(result_item) expected_ap, actual_ap;
  uvm_analysis_port #(bit) reset_ap;
  covergroup behavior with function sample (
      int privilege,
      bit packet_stall,
      bit word_stall,
      bit redirect,
      bit fence,
      bit invalidate,
      int mask,
      bit pf,
      bit af,
      bit executable,
      bit packet_valid,
      bit response_valid
  );
    option.per_instance = 1;
    context_privilege: coverpoint privilege {
      bins user_mode = {0}; bins supervisor = {1}; bins machine_mode = {3};
    }
    cpu_backpressure: coverpoint packet_stall;
    word_backpressure: coverpoint word_stall;
    redirect_event: coverpoint redirect;
    fence_event: coverpoint fence;
    invalidate_event: coverpoint invalidate;
    page_fault: coverpoint pf iff (packet_valid);
    access_fault: coverpoint af iff (packet_valid);
    executable_line: coverpoint executable iff (response_valid);
    packet_mask: coverpoint mask iff (packet_valid) {bins legal[] = `FETCH_MASK_VALUES;}
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    expected_ap = new("expected_ap", this);
    actual_ap = new("actual_ap", this);
    reset_ap = new("reset_ap", this);
    behavior = new();
  endfunction
  function void write(sample_item t);
    result_item expected, actual;
    bit cancel, pf, af, request_fire, response_fire, maintenance_fire, ack_fire, packet_fire;
    logic [`FETCH_PACKET_BITS-1:0] actual_packet;
    longint unsigned expected_npc;
    latest = t;
    if (t.reset) begin
      reset_ap.write(1);
      if (word != null) reset_words++;
      if (maintenance_active) reset_maintenance++;
      offer = null;
      word = null;
      maintenance_active = 0;
      needs_maintenance = 0;
      offered_packet = 0;
      last_valid = 0;
      cursor = `FI(t.inputs, RESETVECTOR);
      epoch++;
      previous_inputs = t.inputs;
      return;
    end
    cycle++;
    expected_npc = `FI(t.inputs, REDIRECT_VALID) ? `FI(t.inputs, REDIRECT_BITS) : cursor;
    evidence::check(`FO(t.outputs, NPC) === expected_npc,
                    "Fetch next-PC output disagrees with accepted CPU packets and redirects");
    // The word descriptor is formed before its physical handshake and remains stable across redirects.
    if (`FO(t.outputs, REQUEST_VALID) && offer == null) begin
      offer = new;
      offer.pc = cursor;
      offer.address = cursor & ~64'h7;
      offer.epoch = epoch;
      evidence::check(
          `FO(t.outputs, REQUEST_BITS_ADDR) === offer.address &&
          `FO(t.outputs, REQUEST_BITS_CONTEXT_PRIVILEGE) === `FI(previous_inputs, CONTEXT_PRIVILEGE)
          && `FO(t.outputs, REQUEST_BITS_CONTEXT_SATP) === `FI(previous_inputs, CONTEXT_SATP) &&
          `FO(t.outputs, REQUEST_BITS_CONTEXT_SUM) === `FI(previous_inputs, CONTEXT_SUM) &&
          `FO(t.outputs, REQUEST_BITS_CONTEXT_MXR) === `FI(previous_inputs, CONTEXT_MXR) &&
          `FO(t.outputs, REQUEST_BITS_EXECUTE) === 1'b1,
          "Fetch word request lost its PC or captured address-space context");
    end
    cancel = `FI(t.inputs, REDIRECT_VALID) || `FI(t.inputs, FLUSH);
    if (cancel) begin
      reset_ap.write(1);
      offered_packet = 0;
      last_valid = 0;
      epoch++;
      if (`FI(t.inputs, REDIRECT_VALID)) cursor = `FI(t.inputs, REDIRECT_BITS);
      evidence::check(!`FO(t.outputs, PACKET_VALID), "Fetch exposed a cancelled CPU packet");
    end
    if (`FI(t.inputs, FLUSH)) needs_maintenance = 1;
    request_fire = `FO(t.outputs, REQUEST_VALID) && `FI(t.inputs, REQUEST_READY);
    response_fire = `FI(t.inputs, RESPONSE_VALID) && `FO(t.outputs, RESPONSE_READY);
    maintenance_fire = `FO(t.outputs, MAINTENANCE_VALID) && `FI(t.inputs, MAINTENANCE_READY);
    ack_fire = `FI(t.inputs, MAINTAINED_VALID) && `FO(t.outputs, MAINTAINED_READY);
    packet_fire = `FO(t.outputs, PACKET_VALID) && `FI(t.inputs, PACKET_READY);
    if (offer != null && needs_maintenance)
      evidence::check(offer.epoch != epoch, "Fetch issued a new word before maintenance completed");
    if (needs_maintenance || maintenance_active)
      evidence::check(!`FO(t.outputs, PACKET_VALID),
                      "Fetch delivered instructions before maintenance completed");
    if (request_fire) begin
      evidence::check(offer != null && word == null && !maintenance_active,
                      "Fetch overlapped backend word transactions");
      word = offer;
      offer = null;
      word.version = image.version;
      word.due = cycle + 2 + reads % 3;
      word.pf = pf_enabled && word.address == pf_address;
      word.af = af_enabled && word.address == af_address;
      word.executable = line_executable && !word.pf && !word.af;
      word.data = word.pf || word.af ? 0 : fetch_ref_word(word.address, word.version);
      word.line = '0;
      if (!word.pf && !word.af)
        for (int index = 0; index < 8; index++)
        word.line[index*64+:64] =
            fetch_ref_word((word.address & ~64'h3f) + index * 8, word.version);
      reads++;
    end
    if (response_fire) begin
      if (word == null) `uvm_fatal("WORD_RESPONSE", "Fetch accepted an unsolicited word")
      evidence::check(`FI(t.inputs, RESPONSE_BITS_DATA) === word.data &&
                      `FI(t.inputs, RESPONSE_BITS_LINE) === word.line &&
                      `FI(t.inputs, RESPONSE_BITS_PAGEFAULT) === word.pf &&
                      `FI(t.inputs, RESPONSE_BITS_ACCESSFAULT) === word.af &&
                      `FI(t.inputs, RESPONSE_BITS_LINEEXECUTABLE) === word.executable,
                      "Fetch backend response disagrees with its programmed image");
      if (word.epoch == epoch) begin
        last_pc = word.pc;
        last_pf = word.pf;
        last_af = word.af;
        last_valid = 1;
      end else drained_words++;
      word = null;
      replies++;
    end
    if (maintenance_fire) begin
      evidence::check(word == null && offer == null && !maintenance_active && needs_maintenance &&
                      `FO(t.outputs, MAINTENANCE_BITS),
                      "Fetch maintenance did not drain its outstanding word");
      maintenance_active = 1;
      maintenance_requests++;
      ack_due = cycle + 4;
    end
    if (ack_fire) begin
      evidence::check(maintenance_active && `FI(t.inputs, MAINTAINED_BITS),
                      "Fetch consumed an invalid maintenance acknowledgement");
      maintenance_active = 0;
      needs_maintenance  = 0;
      maintenance_acks++;
    end
    if (`FI(t.inputs, FLUSH)) needs_maintenance = 1;
    if (!cancel && `FO(t.outputs, PACKET_VALID)) begin
      pf = last_valid && last_pc == cursor ? last_pf : 0;
      af = last_valid && last_pc == cursor ? last_af : 0;
      expected = result_item::type_id::create("expected");
      expected.payload = '0;
      expected.payload[`FETCH_OUT_PACKET_BITS_PC_OFFSET-`FETCH_PACKET_OFFSET+:`FETCH_OUT_PACKET_BITS_PC_WIDTH]=cursor;
      expected.payload[`FETCH_OUT_PACKET_BITS_DATA_OFFSET-`FETCH_PACKET_OFFSET+:`FETCH_OUT_PACKET_BITS_DATA_WIDTH]=pf||af?0:fetch_ref_group(
          cursor, image.version);
      expected.payload[`FETCH_OUT_PACKET_BITS_MASK_OFFSET-`FETCH_PACKET_OFFSET+:`FETCH_OUT_PACKET_BITS_MASK_WIDTH]=`FETCH_INSTRUCTION_BYTES==2?(cursor[1]?2:3):1;
      expected.payload[`FETCH_OUT_PACKET_BITS_PAGEFAULT_OFFSET-`FETCH_PACKET_OFFSET+:`FETCH_OUT_PACKET_BITS_PAGEFAULT_WIDTH]=pf;
      expected.payload[`FETCH_OUT_PACKET_BITS_ACCESSFAULT_OFFSET-`FETCH_PACKET_OFFSET+:`FETCH_OUT_PACKET_BITS_ACCESSFAULT_WIDTH]=af;
      actual_packet = t.outputs[`FETCH_PACKET_OFFSET+:`FETCH_PACKET_BITS];
      if (`FETCH_INSTRUCTION_BYTES == 2 && cursor[1]) begin
        expected.payload[`FETCH_OUT_PACKET_BITS_DATA_OFFSET-`FETCH_PACKET_OFFSET+:16] = 0;
        actual_packet[`FETCH_OUT_PACKET_BITS_DATA_OFFSET-`FETCH_PACKET_OFFSET+:16] = 0;
      end
      if (!offered_packet) begin
        evidence::check(actual_packet === expected.payload, $sformatf(
                        "Fetch offered wrong packet pc=%h expected=%h got=%h",
                        cursor,
                        expected.payload,
                        actual_packet
                        ));
        expected_ap.write(expected);
        offered_packet = 1;
      end
      if (packet_fire) begin
        actual = result_item::type_id::create("actual");
        actual.payload = actual_packet;
        actual_ap.write(actual);
        offered_packet = 0;
        last_valid = 0;
        cursor = (cursor & ~64'h3) + 4;
        checked++;
      end
    end
    behavior.sample(
        `FI(t.inputs, CONTEXT_PRIVILEGE), `FO(t.outputs, PACKET_VALID) && !
        `FI(t.inputs, PACKET_READY), `FO(t.outputs, REQUEST_VALID) && !`FI(t.inputs, REQUEST_READY),
        `FI(t.inputs, REDIRECT_VALID), `FI(t.inputs, FLUSH), `FI(t.inputs, INVALIDATE),
        `FO(t.outputs, PACKET_BITS_MASK), `FO(t.outputs, PACKET_BITS_PAGEFAULT),
        `FO(t.outputs, PACKET_BITS_ACCESSFAULT), `FI(t.inputs, RESPONSE_BITS_LINEEXECUTABLE),
        `FO(t.outputs, PACKET_VALID), response_fire);
    previous_inputs = t.inputs;
  endfunction
  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    evidence::check(word == null && offer == null && !maintenance_active && !needs_maintenance,
                    "Fetch backend did not drain or reset");
    evidence::check(
        reads==replies+reset_words&&maintenance_requests==maintenance_acks+reset_maintenance,
        "Fetch backend transaction accounting mismatch");
    evidence::check(drained_words != 0 && maintenance_acks != 0,
                    "Fetch did not exercise cancelled-word drain and maintenance acknowledgement");
  endfunction
endclass
