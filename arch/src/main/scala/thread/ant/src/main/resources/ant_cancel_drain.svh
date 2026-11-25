// A task that reached ECALL while its accelerator work is still draining is cancelled; a
// transfer requested meanwhile waits for the context. The executor state is observed
// hierarchically to wait for the local end of the task.
task automatic cancel_drain();
  for (int c = 0; c < 2; c++) begin
    put(c * 2, 0, {64'b0, 32'h00000073, 32'h0000307b}, 1);
    entry[c] = 0;
    code_end[c] = 8;
    argument[c] = `TLS_BASE;
    stack[c] = `TLS_END;
    // A launch requested outside the group's lifetime waits for inUse.
    @(negedge clock);
    start_valid[c] = 1;
    repeat (3) begin
      @(negedge clock);
      evidence::check(!running[c], "launch accepted outside the group's lifetime");
    end
    start_valid[c] = 0;
    in_use = 1;
    launch(c);
    drained[c] = 0;
    do
    @(negedge clock);
    while ((c ? dut.ants_1.executor.state : dut.ants_0.executor.state) != 4'hB);
    fork
      copy_local(c, 0, 0, 0, 8);
      begin
        repeat (10) @(negedge clock);
        evidence::check(!result_valid[c] && !transferred_valid[c],
                        "work finished before the drain");
        cancel[c] = 1;
        @(negedge clock);
        cancel[c]  = 0;
        drained[c] = 1;
        finish(c, 0, 1);
      end
    join
    in_use = 0;
  end
endtask
