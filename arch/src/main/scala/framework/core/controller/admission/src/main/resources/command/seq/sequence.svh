class scenario_sequence extends uvm_sequence #(cycle_item);
  `uvm_object_utils(scenario_sequence)
  logic [ `AD_INPUT_BITS-1:0] pins;
  logic [`AD_OUTPUT_BITS-1:0] observed;
  function new(string name = "scenario");
    super.new(name);
  endfunction
  task tick(bit reset = 0);
    cycle_item t = cycle_item::type_id::create("cycle");
    start_item(t);
    t.inputs = pins;
    t.reset  = reset;
    finish_item(t);
    observed = t.outputs;
  endtask
  task command(int tag, logic [31:0] instruction, logic [63:0] rs1, logic [63:0] rs2);
    for (int b = 0; b < `AD_COMMAND_WIDTH; b++)
      pins[`AD_COMMAND_OFFSET+b] = ((b + tag) % 3 == 0) ^ ((b + tag) % 7 == 0);
    pins[`AD_IN_EXECUTION_COMMAND_BITS_INSTRUCTION_RAW_INST_OFFSET+:32]=instruction|((tag%32)<<7);
    pins[`AD_IN_EXECUTION_COMMAND_BITS_INSTRUCTION_RS1DATA_OFFSET+:64] = rs1;
    pins[`AD_IN_EXECUTION_COMMAND_BITS_INSTRUCTION_RS2DATA_OFFSET+:64] = rs2;
    pins[`AD_IN_EXECUTION_COMMAND_BITS_TAG_OFFSET+:`AD_IN_EXECUTION_COMMAND_BITS_TAG_WIDTH] = tag;
    pins[`AD_IN_EXECUTION_COMMAND_VALID_OFFSET] = 1;
  endtask
  task accepted();
    do tick(); while (!observed[`AD_OUT_EXECUTION_COMMAND_READY_OFFSET]);
  endtask
  task body();
    string negative;
    pins = '0;
    repeat (3) tick(1);
    tick();
    if ($value$plusargs("CASE=%s", negative)) begin
      command(0, 0, 0, 0);
      repeat (8) tick();
      `uvm_fatal("MISSING_ASSERTION", "Expected admission assertion did not fire")
    end
    for (int i = 0; i < 48; i++) begin
      bit moving = i % 2;
      bit result = i % 3 != 0;
      command(i, moving ? (32'h1a00007b | (result << 14)) : (32'h2b | (result << 14)),
              i * 64'hfedcba987654321, ~(i * 64'h123456789abcdef));
      pins[`AD_IN_TASKREQUEST_READY_OFFSET]  = 0;
      pins[`AD_IN_MOVE_COMMAND_READY_OFFSET] = 0;
      repeat (3) tick();
      pins[`AD_IN_TASKREQUEST_READY_OFFSET]  = 1;
      pins[`AD_IN_MOVE_COMMAND_READY_OFFSET] = 1;
      accepted();
      pins[`AD_IN_EXECUTION_COMMAND_VALID_OFFSET]  = 0;
      pins[`AD_IN_EXECUTION_RESPONSE_READY_OFFSET] = 0;
      pins[`AD_IN_EXECUTION_COMPLETE_READY_OFFSET] = 0;
      repeat (2) tick();
      if (moving) begin
        pins[`AD_IN_MOVE_COMPLETION_BITS_OFFSET]  = i % 4 >= 2;
        pins[`AD_IN_MOVE_COMPLETION_VALID_OFFSET] = 1;
        do tick(); while (!observed[`AD_OUT_MOVE_COMPLETION_READY_OFFSET]);
        pins[`AD_IN_MOVE_COMPLETION_VALID_OFFSET] = 0;
        if (result) begin
          repeat (3) tick();
          pins[`AD_IN_EXECUTION_RESPONSE_READY_OFFSET] = 1;
          tick();
        end
      end else begin
        pins[`AD_IN_TASKRESPONSE_BITS_RD_OFFSET+:5] = 5'(31 - i);
        pins[`AD_IN_TASKRESPONSE_BITS_DATA_OFFSET+:64] = 64'hcafe000000000000 + i;
        pins[`AD_IN_TASKRESPONSE_VALID_OFFSET] = 1;
        if (result) begin
          repeat (3) tick();
          pins[`AD_IN_EXECUTION_RESPONSE_READY_OFFSET] = 1;
        end
        do tick(); while (!observed[`AD_OUT_TASKRESPONSE_READY_OFFSET]);
        pins[`AD_IN_TASKRESPONSE_VALID_OFFSET] = 0;
      end
      repeat (3) tick();
      pins[`AD_IN_EXECUTION_COMPLETE_READY_OFFSET] = 1;
      do tick(); while (!observed[`AD_OUT_EXECUTION_COMPLETE_VALID_OFFSET]);
      tick();
    end
    command(250, 32'h402b, '1, '1);
    accepted();
    pins[`AD_IN_EXECUTION_COMMAND_VALID_OFFSET] = 0;
    repeat (3) tick();
    repeat (2) tick(1);
    pins = '0;
    tick();
    repeat (4) tick();
  endtask
endclass
