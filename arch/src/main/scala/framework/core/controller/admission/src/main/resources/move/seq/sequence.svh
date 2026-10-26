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
    pins[`AD_IN_ROCC_CMD_BITS_INSTRUCTION_RAW_INST_OFFSET+:32] = instruction | ((tag % 32) << 7);
    pins[`AD_IN_ROCC_CMD_BITS_INSTRUCTION_RS1DATA_OFFSET+:64] = rs1;
    pins[`AD_IN_ROCC_CMD_BITS_INSTRUCTION_RS2DATA_OFFSET+:64] = rs2;
    pins[`AD_IN_ROCC_CMD_VALID_OFFSET] = 1;
  endtask
  task accepted();
    do tick(); while (!observed[`AD_OUT_ROCC_CMD_READY_OFFSET]);
  endtask
  task body();
    string negative;
    pins = '0;
    repeat (3) tick(1);
    tick();
    if ($value$plusargs("CASE=%s", negative)) begin
      command(0, 32'h1a00707b, 0, 0);
      repeat (8) tick();
      `uvm_fatal("MISSING_ASSERTION", "Expected admission assertion did not fire")
    end
    for (int i = 0; i < 48; i++) begin
      command(i, 32'h1a00307b, i * 64'hfedcba987654321,
              i == 47 ? 64'h0000ffff12345678 : ~(i * 64'h123456789abcdef));
      pins[`AD_IN_MOVE_COMMAND_READY_OFFSET] = 0;
      repeat (3) tick();
      pins[`AD_IN_MOVE_COMMAND_READY_OFFSET] = 1;
      accepted();
      pins[`AD_IN_ROCC_CMD_VALID_OFFSET] = 0;
      repeat (5) tick();
      pins[`AD_IN_MOVE_COMPLETION_BITS_OFFSET]  = i % 2;
      pins[`AD_IN_MOVE_COMPLETION_VALID_OFFSET] = 1;
      do tick(); while (!observed[`AD_OUT_MOVE_COMPLETION_READY_OFFSET]);
      pins[`AD_IN_MOVE_COMPLETION_VALID_OFFSET] = 0;
      tick();
    end
    command(0, 32'h1a00307b, '1, '1);
    accepted();
    pins[`AD_IN_ROCC_CMD_VALID_OFFSET] = 0;
    repeat (3) tick();
    repeat (2) tick(1);
    pins = '0;
    tick();
    repeat (4) tick();
  endtask
endclass
