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
    pins[`AD_IN_COMMAND_BITS_INSTRUCTION_RAW_INST_OFFSET+:32] = instruction | ((tag % 32) << 7);
    pins[`AD_IN_COMMAND_BITS_INSTRUCTION_RS1DATA_OFFSET+:64] = rs1;
    pins[`AD_IN_COMMAND_BITS_INSTRUCTION_RS2DATA_OFFSET+:64] = rs2;
    pins[`AD_IN_COMMAND_BITS_TAG_OFFSET+:`AD_IN_COMMAND_BITS_TAG_WIDTH] = tag;
    pins[`AD_IN_COMMAND_VALID_OFFSET] = 1;
  endtask
  task accepted();
    do tick(); while (!observed[`AD_OUT_COMMAND_READY_OFFSET]);
  endtask
  task body();
    string negative;
    pins = '0;
    repeat (3) tick(1);
    tick();
    for (int i = 0; i < 48; i++) begin
      bit remote = i % 2;
      command(i, (32'((remote ? 13 + i % 20 : i % 13) << 25) | 32'h2b), i * 64'h123456789abcdef,
              ~64'(i));
      pins[`AD_IN_LOCAL_REQUEST_READY_OFFSET]  = remote;
      pins[`AD_IN_REMOTE_REQUEST_READY_OFFSET] = !remote;
      repeat (3) tick();
      pins[`AD_IN_LOCAL_REQUEST_READY_OFFSET]  = 1;
      pins[`AD_IN_REMOTE_REQUEST_READY_OFFSET] = 1;
      accepted();
      pins[`AD_IN_COMMAND_VALID_OFFSET] = 0;
      pins[`AD_IN_RESPONSE_READY_OFFSET] = 0;
      pins[`AD_IN_LOCAL_REPLY_BITS_OFFSET+:64] = 64'h9876543200000000 + i;
      pins[`AD_IN_REMOTE_REPLY_BITS_OFFSET+:64] = 64'h1234567800000000 + i;
      pins[`AD_IN_LOCAL_REPLY_VALID_OFFSET] = remote;
      pins[`AD_IN_REMOTE_REPLY_VALID_OFFSET] = !remote;
      repeat (3) tick();
      pins[`AD_IN_LOCAL_REPLY_VALID_OFFSET]  = 1;
      pins[`AD_IN_REMOTE_REPLY_VALID_OFFSET] = 1;
      repeat (3) tick();
      pins[`AD_IN_RESPONSE_READY_OFFSET] = 1;
      do tick(); while (!observed[`AD_OUT_RESPONSE_VALID_OFFSET]);
      pins[`AD_IN_LOCAL_REPLY_VALID_OFFSET]  = 0;
      pins[`AD_IN_REMOTE_REPLY_VALID_OFFSET] = 0;
      tick();
    end
    command(0, 32'h2b, '1, '1);
    accepted();
    pins[`AD_IN_COMMAND_VALID_OFFSET] = 0;
    repeat (3) tick();
    repeat (2) tick(1);
    pins = '0;
    tick();
    repeat (4) tick();
  endtask
endclass
