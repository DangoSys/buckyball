class write_sequence extends uvm_sequence #(item);
  `uvm_object_utils(write_sequence)
  int unsigned beats;
  bit masked;
  function new(string name = "write_sequence");
    super.new(name);
  endfunction
  task body();
    item req;
    for (int i = 0; i < beats; i++) begin
      req = item::type_id::create("req");
      start_item(req);
      req.data = 32'h9e3779b9 * (i + 1);
      req.keep = masked ? (i % 16) : 15;
      req.last = i == beats - 1;
      finish_item(req);
    end
  endtask
endclass

class command_sequence extends uvm_sequence #(command);
  `uvm_object_utils(command_sequence)
  command req;
  function new(string name = "command_sequence");
    super.new(name);
  endfunction
  task body();
    start_item(req);
    finish_item(req);
  endtask
endclass

class access_sequence extends uvm_sequence;
  `uvm_object_utils(access_sequence)
  env test_env;
  int unsigned reads = 0;
  int unsigned commands = 0;
  function new(string name = "access_sequence");
    super.new(name);
  endfunction
  task transfer(bit write, int unsigned word_addr, int unsigned beats, bit masked = 0,
                int unsigned stall = 0);
    command_sequence cmd = command_sequence::type_id::create("command");
    write_sequence   data = write_sequence::type_id::create("data");
    cmd.req = command::type_id::create("req");
    cmd.req.write = write;
    cmd.req.word_addr = word_addr;
    cmd.req.beats = beats;
    cmd.req.completion_stall = stall;
    data.beats = beats;
    data.masked = masked;
    fork
      cmd.start(test_env.commands.seqr);
      if (write) data.start(test_env.source.seqr);
    join
    commands++;
    if (!write) begin
      reads += beats;
      test_env.scoreboard.wait_checked(reads);
    end
  endtask
  task body();
    test_env.reset.pulse(4);
    transfer(1, 0, `BANK_WORDS);
    test_env.sink.stall_cycles = 0;
    transfer(0, 0, `BANK_WORDS);
    test_env.sink.stall_cycles = 3;
    for (int b = 0; b < `BANK_COUNT; b++) begin
      transfer(1, b, `BANK_COUNT + 1, 1, 3);
      transfer(0, b, `BANK_COUNT + 1, 0, 3);
    end
    transfer(1, `BANK_WORDS - 1, 1);
    transfer(0, `BANK_WORDS - 1, 1);
    test_env.reset.pulse(2);
    transfer(0, 0, `BANK_WORDS, 0, 3);
  endtask
endclass
