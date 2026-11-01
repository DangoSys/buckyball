// Reset while a task waits on its accelerator command releases the group and the context;
// a new group and task follow.
task automatic service_reset();
  put(0, 0, {64'b0, 32'h00000073, 32'h0000307b});
  command(5, 0, 0);
  command_ready[0] = 0;
  launch(0, 9);
  wait (command_valid[0]);
  @(negedge clock);
  reset = 1;
  repeat (2) @(negedge clock);
  reset = 0;
  command_ready[0] = 1;
  if (command_valid) `uvm_fatal("RESET", "reset left an accelerator command behind")
  put(0, 0, {64'b0, 32'h00000073, 32'h02500513});
  command(5, 0, 0);
  launch(0, 10);
  finish(0, 10, 37);
  command(6, 0, 0);
endtask
