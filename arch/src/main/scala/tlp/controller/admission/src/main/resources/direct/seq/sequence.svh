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
      pins[`AD_IN_ACCELERATOR_CMD_READY_OFFSET] = 1;
      if (negative == "reply") pins[`AD_IN_ACCELERATOR_RESP_VALID_OFFSET] = 1;
      else if (negative == "interrupt") pins[`AD_IN_ACCELERATOR_INTERRUPT_OFFSET] = 1;
      else command(0, 32'h407b, 0, 0);
      repeat (8) tick();
      `uvm_fatal("MISSING_ASSERTION", "Expected admission assertion did not fire")
    end
    pins[`AD_IN_ACCELERATOR_CMD_READY_OFFSET] = 1;
    pins[`AD_IN_EXECUTION_COMPLETE_READY_OFFSET] = 0;
    pins[`AD_IN_ACCELERATOR_BUSY_OFFSET] = 1;
    for (int i = 0; i < `AD_ENTRIES; i++) begin
      command(i, 32'h7b, i * 64'h123456789abcdef, ~64'(i));
      accepted();
      pins[`AD_IN_EXECUTION_COMMAND_VALID_OFFSET] = 0;
    end
    command(100, 32'h7b, '1, '1);
    repeat (5) tick();
    pins[`AD_IN_EXECUTION_COMMAND_VALID_OFFSET] = 0;
    pins[`AD_IN_ACCELERATOR_BUSY_OFFSET] = 0;
    pins[`AD_IN_EXECUTION_COMPLETE_READY_OFFSET] = 1;
    do tick(); while (!observed[`AD_OUT_DRAINED_OFFSET]);
    for (int i = 0; i < 32; i++) begin
      bit result = i % 2;
      pins[`AD_IN_REMOTE_BUSY_OFFSET] = 1;
      command(16 + i, (32'((13 + i % 20) << 25) | 32'h2b | (result << 14)), i * 64'h123456789abcdef,
              ~64'(i));
      repeat (3) tick();
      pins[`AD_IN_REMOTE_BUSY_OFFSET] = 0;
      pins[`AD_IN_REMOTE_REQUEST_READY_OFFSET] = 0;
      repeat (3) tick();
      pins[`AD_IN_REMOTE_REQUEST_READY_OFFSET] = 1;
      accepted();
      pins[`AD_IN_EXECUTION_COMMAND_VALID_OFFSET] = 0;
      pins[`AD_IN_EXECUTION_RESPONSE_READY_OFFSET] = 0;
      pins[`AD_IN_EXECUTION_COMPLETE_READY_OFFSET] = 0;
      pins[`AD_IN_REMOTE_REPLY_BITS_OFFSET+:64] = 64'h1234000056780000 + i;
      pins[`AD_IN_REMOTE_REPLY_VALID_OFFSET] = 1;
      if (result) begin
        repeat (3) tick();
        pins[`AD_IN_EXECUTION_RESPONSE_READY_OFFSET] = 1;
      end
      do tick(); while (!observed[`AD_OUT_REMOTE_REPLY_READY_OFFSET]);
      pins[`AD_IN_REMOTE_REPLY_VALID_OFFSET] = 0;
      repeat (3) tick();
      pins[`AD_IN_EXECUTION_COMPLETE_READY_OFFSET] = 1;
      do tick(); while (!observed[`AD_OUT_EXECUTION_COMPLETE_VALID_OFFSET]);
      tick();
    end
    pins[`AD_IN_EXECUTION_COMPLETE_READY_OFFSET] = 0;
    command(200, 32'h7b, '1, '1);
    accepted();
    pins[`AD_IN_EXECUTION_COMMAND_VALID_OFFSET] = 0;
    repeat (3) tick();
    repeat (2) tick(1);
    pins = '0;
    tick();
    repeat (4) tick();
  endtask
endclass
