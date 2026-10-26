class scenario_sequence extends uvm_sequence;
  `uvm_object_utils(scenario_sequence)
  env test_env;
  virtual fetch_if vif;
  function new(string name = "scenario_sequence");
    super.new(name);
  endfunction
  task operate(control_kind kind, longint unsigned target = 0, int version = 0, int privilege = 3,
               longint unsigned satp = 0, bit sum = 0, bit mxr = 0);
    control_sequence seq = control_sequence::type_id::create("control");
    seq.item = control_item::type_id::create("item");
    seq.item.kind = kind;
    seq.item.target = target;
    seq.item.version = version;
    seq.item.privilege = privilege;
    seq.item.satp = satp;
    seq.item.sum = sum;
    seq.item.mxr = mxr;
    seq.start(test_env.seqr);
  endtask
  task packet_at(longint unsigned target);
    do @(negedge vif.clock); while (!`FO(test_env.reference.latest.outputs, PACKET_VALID));
    evidence::check(`FO(test_env.reference.latest.outputs, PACKET_BITS_PC) === target,
                    "Fetch packet appeared at the wrong redirected PC");
  endtask
  task consume(int count);
    int target = test_env.reference.checked + count;
    test_env.reference.packet_ready = 1;
    while (test_env.reference.checked < target) @(negedge vif.clock);
    test_env.reference.packet_ready = 0;
    @(negedge vif.clock);
  endtask
  task body();
    int baseline, target;
    longint unsigned high;
    operate(RESET_VECTOR, 'h1000);
    test_env.reset.pulse(4);
    packet_at('h1000);
    consume(64);
    // Fill both line buffers, then alternate held packets without a backend read.
    operate(FENCE, 'h1000);
    packet_at('h1000);
    operate(REDIRECT, 'h1040);
    packet_at('h1040);
    baseline = test_env.reference.reads;
    for (int n = 0; n < 8; n++) begin
      operate(REDIRECT, n % 2 ? 'h1040 : 'h1000);
      packet_at(n % 2 ? 'h1040 : 'h1000);
      evidence::check(test_env.reference.reads == baseline,
                      "Fetch did not retain two executable lines");
    end
    operate(REDIRECT, 'h1080);
    packet_at('h1080);
    baseline = test_env.reference.reads;
    operate(REDIRECT, 'h1040);
    packet_at('h1040);
    evidence::check(test_env.reference.reads == baseline,
                    "Fetch replaced its most recently used line");
    operate(REDIRECT, 'h1000);
    packet_at('h1000);
    evidence::check(test_env.reference.reads == baseline + 1,
                    "Fetch did not replace its least recently used line");
    // Every individual context field participates in the line-buffer key.
    for (int field = 0; field < 4; field++) begin
      operate(CONTEXT, 'h1000, 0, 3, 0, 0, 0);
      operate(FENCE, 'h1000);
      packet_at('h1000);
      baseline = test_env.reference.reads;
      operate(CONTEXT, 'h1000, 0, field == 0 ? 1 : 3, field == 1 ? 64'h8000000000001234 : 0,
              field == 2, field == 3);
      packet_at('h1000);
      evidence::check(test_env.reference.reads == baseline + 1,
                      "Fetch ignored an address-space context field");
      operate(CONTEXT, 'h1000, 0, 3, 0, 0, 0);
      packet_at('h1000);
      evidence::check(test_env.reference.reads == baseline + 1,
                      "Fetch lost the retained prior context");
    end
    baseline = test_env.reference.reads;
    operate(INVALIDATE);
    operate(REDIRECT, 'h1000);
    packet_at('h1000);
    evidence::check(test_env.reference.reads == baseline + 1,
                    "Fetch did not invalidate its buffered line");
    // Redirects drain stalled offers and accepted words instead of cancelling the backend port.
    test_env.reference.hold_request = 1;
    operate(REDIRECT, 'h2000);
    wait (`FO(test_env.reference.latest.outputs, REQUEST_VALID));
    operate(REDIRECT, 'h3000);
    repeat (4) @(negedge vif.clock);
    test_env.reference.hold_request = 0;
    packet_at('h3000);
    test_env.reference.hold_word = 1;
    operate(REDIRECT, 'h4000);
    wait (test_env.reference.word != null);
    operate(REDIRECT, 'h5000);
    repeat (4) @(negedge vif.clock);
    test_env.reference.hold_word = 0;
    packet_at('h5000);
    // An image replacement and fence occur atomically; the older response must not refill either buffer.
    test_env.reference.hold_word = 1;
    operate(REDIRECT, 'h6000);
    wait (test_env.reference.word != null);
    test_env.reference.hold_maintenance = 1;
    test_env.reference.hold_ack = 1;
    operate(FENCE, 'h7000, 1);
    repeat (4) @(negedge vif.clock);
    evidence::check(!`FO(test_env.reference.latest.outputs, MAINTENANCE_VALID),
                    "Fetch maintenance overtook the outstanding word");
    test_env.reference.hold_word = 0;
    wait (`FO(test_env.reference.latest.outputs, MAINTENANCE_VALID));
    repeat (4) @(negedge vif.clock);
    test_env.reference.hold_maintenance = 0;
    wait (test_env.reference.maintenance_active);
    repeat (4) @(negedge vif.clock);
    test_env.reference.hold_ack = 0;
    packet_at('h7000);
    operate(REDIRECT, 'h6000);
    packet_at('h6000);
    consume(32);
    // Fault packets carry no instruction data and are never buffered.
    test_env.reference.line_executable = 0;
    test_env.reference.pf_enabled = 1;
    test_env.reference.af_enabled = 1;
    test_env.reference.pf_address = 'ha000;
    test_env.reference.af_address = 'ha008;
    operate(FENCE, 'ha000, 1);
    packet_at('ha000);
    consume(6);
    test_env.reference.pf_enabled = 0;
    test_env.reference.af_enabled = 0;
    if (`FETCH_INSTRUCTION_BYTES == 2) begin
      operate(FENCE, 'hb002, 1);
      packet_at('hb002);
      consume(12);
    end
    // Exercise every virtual-address bit without interpreting instructions in this IP.
    test_env.reference.line_executable = 1;
    for (int bit_index = 2; bit_index < 64; bit_index++) begin
      high = 64'(1) << bit_index;
      operate(FENCE, high, bit_index + 2);
      packet_at(high);
      consume(1);
    end
    // SATP mode, PPN and ASID bits are independently part of the buffered context.
    for (int bit_index = 0; bit_index < 60; bit_index++) begin
      operate(CONTEXT, 'h1000, 0, 1, 64'h8000000000000000);
      packet_at('h1000);
      baseline = test_env.reference.reads;
      operate(CONTEXT, 'h1000, 0, 1, 64'h8000000000000000 | (64'(1) << bit_index));
      packet_at('h1000);
      evidence::check(test_env.reference.reads == baseline + 1, "Fetch ignored a SATP context bit");
      operate(CONTEXT, 'h1000, 0, 1, 64'h8000000000000000);
      packet_at('h1000);
      evidence::check(test_env.reference.reads == baseline + 1,
                      "Fetch did not retain its baseline SATP context");
    end
    operate(CONTEXT, 'h1000, 0, 0, 0);
    packet_at('h1000);
    consume(2);
    // Reset at offer, receive and held CPU-packet boundaries.
    for (int phase_index = 0; phase_index < 3; phase_index++) begin
      test_env.reference.hold_request = phase_index == 0;
      test_env.reference.hold_word = phase_index == 1;
      operate(REDIRECT, 'hc000 + phase_index * 'h1000);
      if (phase_index == 0) wait (`FO(test_env.reference.latest.outputs, REQUEST_VALID));
      else if (phase_index == 1) wait (test_env.reference.word != null);
      else packet_at('he000);
      test_env.reset.pulse(3);
      test_env.reference.hold_request = 0;
      test_env.reference.hold_word = 0;
      packet_at('h1000);
      consume(2);
    end
    test_env.reference.hold_ack = 1;
    operate(FENCE, 'hf000, test_env.image.version);
    wait (test_env.reference.maintenance_active);
    test_env.reset.pulse(3);
    test_env.reference.hold_ack = 0;
    packet_at('h1000);
    consume(2);
    test_env.reference.hold_maintenance = 1;
    operate(FENCE, 'hf000, test_env.image.version);
    wait (`FO(test_env.reference.latest.outputs, MAINTENANCE_VALID));
    test_env.reset.pulse(3);
    test_env.reference.hold_maintenance = 0;
    packet_at('h1000);
    consume(2);
    test_env.reference.packet_ready = 0;
    test_env.reset.pulse(3);
    `uvm_info("FETCH",
              $sformatf(
                  "packets=%0d backend_reads=%0d replies=%0d redirected_drains=%0d maintenance=%0d",
                  test_env.reference.checked, test_env.reference.reads, test_env.reference.replies,
                  test_env.reference.drained_words, test_env.reference.maintenance_acks), UVM_LOW)
  endtask
endclass
class invalid_sequence extends uvm_sequence;
  `uvm_object_utils(invalid_sequence)
  reset_controller reset;
  uvm_sequencer #(control_item) seqr;
  function new(string name = "invalid_sequence");
    super.new(name);
  endfunction
  task command(control_kind kind, longint unsigned target = 0, int privilege = 3);
    control_sequence seq = control_sequence::type_id::create("command");
    seq.item = control_item::type_id::create("item");
    seq.item.kind = kind;
    seq.item.target = target;
    seq.item.privilege = privilege;
    seq.item.version = 0;
    seq.item.satp = 0;
    seq.item.sum = 0;
    seq.item.mxr = 0;
    seq.start(seqr);
  endtask
  task body();
    string scenario;
    if (!$value$plusargs("CASE=%s", scenario))
      `uvm_fatal("CASE", "Fetch assertion scenario is required")
    if (scenario == "reset") command(RESET_VECTOR, 1);
    reset.pulse(4);
    case (scenario)
      "reset": begin
      end
      "redirect": command(REDIRECT, 1);
      "flush": command(BAD_FLUSH);
      "privilege": command(CONTEXT, 'h1000, 2);
      "ack": command(FENCE, 0);
      default: `uvm_fatal("CASE", "Unknown fetch assertion scenario")
    endcase
    #1000ns;
    `uvm_fatal("EXPECTED_ASSERTION", "Invalid fetch input did not trigger its assertion")
  endtask
endclass
