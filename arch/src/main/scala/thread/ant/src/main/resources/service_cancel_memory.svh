// A cancel request sent while a load waits on TLS or TSS: the accepted access drains and
// its response is discarded. If the load finished first the task completes normally. The
// executor state is observed hierarchically to place the request.
task automatic cancel_memory();
  for (int n = 0; n < 8; n++) begin
    int c = n % 2;
    bit shared = (n / 2) % 2;
    longint unsigned status, value;
    // nop; lui t0, TLS or TSS base; ld a0, 0(t0); ecall
    put(c * 2, 0, {32'h00000073, 32'h0002b503, shared ? 32'h000202b7 : 32'h000102b7, 32'h00000013});
    command(5, 0, 0);
    command(1, c, 0, 20 + n);
    command(1, c, 1, 0);
    command(1, c, 2, 16);
    command(1, c, 3, `TLS_BASE);
    command(1, c, 4, `TLS_END);
    command(1, c, 5, c ? `SIGNATURE_1 : `SIGNATURE_0);
    command(2, c, 0);
    do @(negedge clock); while ((c ? dut.ant_1.executor.state : dut.ant.executor.state) != 4'h4);
    command(4, c, 0);
    do request(0, c, 1, 0, status); while (!(status & 4));
    value = shared ? 64'h3333 : (c ? 64'h2222 : 64'h1111);
    finish(c, 20 + n, status == 13 ? 0 : value, status == 13);
    command(6, 0, 0);
  end
endtask
