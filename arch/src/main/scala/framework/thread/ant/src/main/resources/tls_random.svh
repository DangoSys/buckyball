// Random traffic on one TLS store: every size code (legal and illegal), aligned, misaligned
// and out-of-window addresses, masks beyond the access width, reads carrying masks, payload
// changes while no request is valid, ownership switches, response backpressure and
// cancellation or reset of accepted accesses. Every response is checked by the SPM model.
task automatic tls_random(int core, int count);
  for (int n = 0; n < count; n++) begin
    virtual spm_if port;
    logic [63:0] address;
    logic [2:0] size;
    logic [15:0] mask;
    logic [127:0] value;
    bit write;
    int shape, bytes;
    // Payload changes while valid is low must not reach the SRAM.
    for (int i = 0; i < 4; i++) begin
      ports[core*2+i%2].address = {$urandom, $urandom};
      ports[core*2+i%2].data = {$urandom, $urandom, $urandom, $urandom};
      ports[core*2+i%2].mask = $urandom;
      ports[core*2+i%2].size = $urandom;
      ports[core*2+i%2].write = $urandom;
    end
    if ($urandom_range(0, 7) == 0) begin
      @(negedge clock);
      active[core] = !active[core];
    end
    port = ports[core*2+active[core]];
    size = $urandom_range(0, 9) < 8 ? $urandom_range(0, 4) : $urandom_range(5, 7);
    bytes = size <= 4 ? 1 << size : 16;
    shape = $urandom_range(0, 15);
    address = `LOCAL_BASE + ($urandom_range(0, `LOCAL_BYTES - 1) & ~(bytes - 1));
    if (shape == 0) address = address | 1;
    if (shape == 1) address = `LOCAL_BASE - bytes;
    if (shape == 2) address = `LOCAL_BASE + `LOCAL_BYTES - (size > 4 ? 1 : 0);
    if (shape == 3) address = {$urandom, $urandom};
    if (shape == 4) address = {$urandom, $urandom} | 64'h8000000000000000;
    write = $urandom;
    mask  = $urandom_range(0, 3) == 0 ? $urandom : (17'd1 << bytes) - 1;
    value = {$urandom, $urandom, $urandom, $urandom};
    port.send(address, size, write, value, mask);
    if (active[core] && $urandom_range(0, 5) == 0) begin
      // Cancel an accepted thread access; its response must never appear.
      repeat ($urandom_range(0, 2)) @(negedge clock);
      port.cancel = 1;
      @(negedge clock);
      port.cancel = 0;
      repeat (3) begin
        @(negedge clock);
        if (port.response_valid) `uvm_fatal("CANCEL", "cancelled TLS response survived")
      end
    end else if (shape == 5) begin
      // Reset cancels the accepted access; SRAM contents and the model remain.
      repeat ($urandom_range(0, 1)) @(negedge clock);
      reset[core] = 1;
      repeat ($urandom_range(1, 2)) @(negedge clock);
      reset[core] = 0;
      if (port.response_valid) `uvm_fatal("RESET", "TLS response survived reset")
    end else begin
      repeat ($urandom_range(0, 3)) @(negedge clock);
      port.take(value);
    end
  end
endtask
