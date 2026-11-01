// Cancel while a load waits on TLS or TSS: the accepted access drains, its response is
// discarded and the context completes as cancelled. The executor state is observed
// hierarchically to place the cancel in the wait cycle.
task automatic cancel_memory();
  for (int n = 0; n < 8; n++) begin
    int c = n % 2;
    bit shared = (n / 2) % 2;
    // nop; lui t0, TLS or TSS base; ld a0, 0(t0); ecall
    put(c * 2, 0, {32'h00000073, 32'h0002b503, shared ? 32'h000202b7 : 32'h000102b7, 32'h00000013},
        1);
    entry[c] = 0;
    code_end[c] = 16;
    argument[c] = `TLS_BASE;
    stack[c] = `TLS_END;
    in_use = 1;
    launch(c);
    do
    @(negedge clock);
    while ((c ? dut.ants_1.executor.state : dut.ants_0.executor.state) != 4'h5);
    cancel[c] = 1;
    @(negedge clock);
    cancel[c] = 0;
    finish(c, 0, 1);
    in_use = 0;
  end
endtask
