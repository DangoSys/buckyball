// Reset while a context executes, and again while a finished task's result waits to be
// taken, drops the task; an unsolicited response or fault during reset is ignored. Private
// code and data persist and the context launches again.
task automatic ant_reset();
  put(0, 0, {32'h00000073, 32'h02b54533, 32'h00053503, 32'h00853583}, 1);
  put(1, `TLS_BASE, {64'd37, 64'h7fffffffffffffff}, 1);
  entry[0] = 0;
  code_end[0] = 16;
  argument[0] = `TLS_BASE;
  stack[0] = `TLS_END;
  in_use = 1;
  for (int round = 0; round < 2; round++) begin
    launch(0);
    if (round == 0) repeat (6) @(negedge clock);
    else wait (result_valid[0]);
    reset = 1;
    response_valid = '1;
    interrupt = '1;
    repeat (2) @(negedge clock);
    response_valid = 0;
    interrupt = 0;
    reset = 0;
    evidence::check(!running && !result_valid && !transferred_valid && !command_valid,
                    "reset left work behind");
  end
  launch(0);
  finish(0, ant_arithmetic(64'h7fffffffffffffff, 37, 4, 0));
  in_use = 0;
endtask
