class model extends uvm_subscriber #(cycle_item);
  `uvm_component_utils(model)
  int ball_ids[`BOOT_BALL_COUNT] = `BOOT_BALL_IDS;
  bit start_epoch = 1, waiting = 0, runtime_phase = 0;
  int seen = 0;
  uvm_analysis_port #(result_item) expected_ap, actual_ap;
  uvm_analysis_port #(bit) reset_ap;
  covergroup behavior with function sample (
      bit runtime_command, int target, bit input_stall, bit output_stall, bit scheduler_idle
  );
    option.per_instance = 1;
    command_kind: coverpoint runtime_command;
    initialization_target: coverpoint target iff (!runtime_command) {
      bins selected[] = `BOOT_ID_BINS;
    }
    source_backpressure: coverpoint input_stall;
    sink_backpressure: coverpoint output_stall;
    scheduler_drained: coverpoint scheduler_idle;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    expected_ap = new("expected_ap", this);
    actual_ap = new("actual_ap", this);
    reset_ap = new("reset_ap", this);
    behavior = new();
  endfunction
  function void write(cycle_item t);
    result_item expected, actual;
    bit input_fire, output_fire, scheduler_idle;
    if (t.reset) begin
      reset_ap.write(1);
      start_epoch = 1;
      waiting = 0;
      runtime_phase = 0;
      seen = 0;
      return;
    end
    if (start_epoch) begin
      foreach (ball_ids[i]) begin
        expected = result_item::type_id::create("initialization");
        expected.payload = '0;
        expected.payload[`BOOT_OUT_CMD_BITS_INSTRUCTION_RAW_INST_OFFSET-`BOOT_OUT_PACKET_OFFSET+:32] = 32'h0a00007b;
        expected.payload[`BOOT_OUT_CMD_BITS_INSTRUCTION_RS2DATA_OFFSET-`BOOT_OUT_PACKET_OFFSET+:`BOOT_OUT_CMD_BITS_INSTRUCTION_RS2DATA_WIDTH] = ball_ids[i];
        expected_ap.write(expected);
      end
      start_epoch = 0;
    end
    input_fire = `BI(t.inputs, COMMAND_VALID) && `BO(t.outputs, COMMAND_READY);
    output_fire = `BO(t.outputs, CMD_VALID) && `BI(t.inputs, CMD_READY);
    scheduler_idle = `BI(t.inputs, SCHEDULERIDLE);
    if (seen < `BOOT_BALL_COUNT)
      evidence::check(!`BO(t.outputs, COMMAND_READY),
                      "Boot accepted runtime input before initialization ended");
    if (waiting) begin
      if (seen < `BOOT_BALL_COUNT || !scheduler_idle)
        evidence::check(!`BO(t.outputs, CMD_VALID),
                        "Boot issued a command before prior work drained");
      if (scheduler_idle) waiting = 0;
    end
    if (seen == `BOOT_BALL_COUNT && !waiting && (
        `BO(t.outputs, COMMAND_READY)
        ||
        `BO(t.outputs, IDLE)
        ))
      runtime_phase = 1;
    if (runtime_phase) begin
      evidence::check(`BO(t.outputs, COMMAND_READY) === `BI(t.inputs, CMD_READY),
                      "Boot runtime ready did not pass through");
      evidence::check(`BO(t.outputs, CMD_VALID) === `BI(t.inputs, COMMAND_VALID),
                      "Boot runtime valid did not pass through");
    end
    if (input_fire) begin
      evidence::check(seen == `BOOT_BALL_COUNT && !waiting,
                      "Boot forwarded a runtime command before initialization drained");
      expected = result_item::type_id::create("runtime");
      expected.payload = t.inputs[`BOOT_IN_PACKET_OFFSET+:`BOOT_PACKET_BITS];
      expected_ap.write(expected);
    end
    if (output_fire) begin
      actual = result_item::type_id::create("actual");
      actual.payload = t.outputs[`BOOT_OUT_PACKET_OFFSET+:`BOOT_PACKET_BITS];
      if (seen < `BOOT_BALL_COUNT) begin
        behavior.sample(0, ball_ids[seen], `BI(t.inputs, COMMAND_VALID) && !
                        `BO(t.outputs, COMMAND_READY), 0, scheduler_idle);
        seen++;
        waiting = 1;
      end else begin
        evidence::check(input_fire, "Boot produced an unsolicited runtime output");
        behavior.sample(1, 0, 0, 0, scheduler_idle);
      end
      actual_ap.write(actual);
    end
    if (!output_fire)
      behavior.sample(1, 0, `BI(t.inputs, COMMAND_VALID) && !`BO(t.outputs, COMMAND_READY),
                      `BO(t.outputs, CMD_VALID) && !`BI(t.inputs, CMD_READY), scheduler_idle);
    if (`BO(t.outputs, IDLE))
      evidence::check(seen == `BOOT_BALL_COUNT && scheduler_idle && !`BI(t.inputs, COMMAND_VALID)
                      && !`BO(t.outputs, CMD_VALID), "Boot declared idle before all work drained");
  endfunction
endclass
