// Random RV64IM programs on both contexts in lockstep with the reference hart. Each program
// uses a random 1 KiB TLS window and a random 1 KiB TSS window of its context, loads,
// stores, ALU/word/M operations, register corner values, forward branches and jumps, raw
// accelerator commands under random readiness and drain, fences and `csrr mcycle`. Code
// sits at a random entry; launches carry random descriptors and task contexts and may
// queue behind the code transfer. Some tasks are cancelled at a random point; the others
// return a0 and their TLS window, TSS window and code are read back through the TSS.
task automatic host_check(longint unsigned address, longint unsigned low, longint unsigned high,
                          string what);
  logic [127:0] word;
  host_access(address, 0, 0, word);
  evidence::check(word === {high, low}, $sformatf(
                  "%s at %h: %h expected %h", what, address, word, {high, low}));
endtask

task automatic programs(int c, int count);
  for (int n = 0; n < count; n++) begin
    int unsigned code[$];
    longint unsigned tls_window = `TLS_BASE + $urandom_range(0, 15) * 1024;
    longint unsigned tss_window = `TSS_BASE + c * 4096 + $urandom_range(0, 3) * 1024;
    int staging = 'h2000 + c * 'h1000;
    int origin, bytes;
    logic [127:0] word, ignored;
    bit cancelled = 0;
    longint unsigned expected;
    generate_program($urandom_range(10, 150), tls_window, tss_window, code);
    origin = $urandom_range(0, (`CODE_BYTES - 4 * code.size() - 16) / 8) * 8;
    bytes  = (4 * code.size() + 15) & ~15;
    while (4 * code.size() < bytes) code.push_back($urandom);
    for (int offset = 0; offset < 1024; offset += 16) begin
      word = {$urandom, $urandom, $urandom, $urandom};
      host_access(`TSS_BASE + staging + offset, 1, word, ignored);
      ant_hart_store(harts[c], tls_window + offset, word[63:0]);
      ant_hart_store(harts[c], tls_window + offset + 8, word[127:64]);
      word = {$urandom, $urandom, $urandom, $urandom};
      host_access(tss_window + offset, 1, word, ignored);
      ant_hart_store(harts[c], tss_window + offset, word[63:0]);
      ant_hart_store(harts[c], tss_window + offset + 8, word[127:64]);
    end
    move(c, 0, 1, tls_window - `TLS_BASE, staging, 1024);
    for (int i = 0; i < code.size(); i += 4) begin
      host_access(`TSS_BASE + staging + 4 * i, 1, {code[i+3], code[i+2], code[i+1], code[i]},
                  ignored);
      for (int k = 0; k < 4; k++) ant_hart_code(harts[c], origin + 4 * (i + k), code[i+k]);
    end
    task_id[c] = $urandom;
    entry[c] = origin;
    code_end[c] = origin + 4 * code.size();
    argument[c] = `TLS_BASE + $urandom_range(0, `TLS_BYTES - 1);
    stack[c] = `TLS_BASE + 16 * $urandom_range(1, `TLS_BYTES / 16);
    start_satp[c] = {$urandom, $urandom};
    start_privilege[c] = $urandom;
    start_sum[c] = $urandom;
    start_mxr[c] = $urandom;
    for (int i = 0; i < `PMP_BITS; i += 32) start_pmp[c][i+:32] = $urandom;
    ant_hart_start(harts[c], entry[c], argument[c], stack[c]);
    chaos[c] = 1;
    if ($urandom_range(0, 3) == 0) begin
      // Start queues behind the code transfer of the same context.
      fork
        move(c, 1, 1, origin, staging, bytes);
        begin
          repeat (2) @(negedge clock);
          checking[c] = 1;
          launch(c);
        end
      join
    end else begin
      move(c, 1, 1, origin, staging, bytes);
      checking[c] = 1;
      launch(c);
    end
    // The descriptor is meaningless while no launch is requested: random values, or a
    // descriptor that would be rejected for exactly one reason.
    entry[c] = {$urandom, $urandom};
    code_end[c] = {$urandom, $urandom};
    argument[c] = {$urandom, $urandom};
    stack[c] = {$urandom, $urandom};
    if (n % 2) begin
      entry[c] = 0;
      code_end[c] = 16;
      argument[c] = `TLS_BASE;
      stack[c] = `TLS_END;
      case ((n / 2) % 9)
        0: entry[c] = 2;
        1: code_end[c] = 18;
        2: code_end[c] = `CODE_BYTES + 8;
        3: entry[c] = 16;
        4: argument[c] = `TLS_BASE - 8;
        5: argument[c] = `TLS_END;
        6: stack[c] = `TLS_BASE;
        7: stack[c] = `TLS_END + 16;
        8: stack[c] = `TLS_END - 8;
      endcase
    end
    if ($urandom_range(0, 4) == 0) begin
      repeat ($urandom_range(0, 300)) @(negedge clock);
      cancel[c]   = 1;
      checking[c] = 0;
      @(posedge clock);
      // A completion published before the cancel is preserved.
      cancelled = !result_valid[c];
      @(negedge clock);
      cancel[c] = 0;
    end
    wait (result_valid[c]);
    chaos[c] = 0;
    drained[c] = 1;
    command_ready[c] = 1;
    if (!cancelled) begin
      longint unsigned pc, data;
      int unsigned instruction, rd;
      checking[c] = 0;
      evidence::check(ant_hart_step(harts[c], 0, pc, instruction, rd, data) == 1, $sformatf(
                      "context %0d: task ended away from its ECALL at %h", c, pc));
      expected = data;
    end
    finish(c, cancelled ? 0 : expected, cancelled);
    if (!cancelled) begin
      for (int offset = 0; offset < 1024; offset += 16)
      host_check(tss_window + offset, ant_hart_load(harts[c], tss_window + offset), ant_hart_load(
                 harts[c], tss_window + offset + 8), "TSS window");
      move(c, 0, 0, tls_window - `TLS_BASE, staging, 1024);
      for (int offset = 0; offset < 1024; offset += 16)
      host_check(`TSS_BASE + staging + offset, ant_hart_load(harts[c], tls_window + offset),
                 ant_hart_load(harts[c], tls_window + offset + 8), "TLS window");
      move(c, 1, 0, origin, staging, bytes);
      for (int i = 0; i < code.size(); i += 4)
      host_check(`TSS_BASE + staging + 4 * i, {code[i+1], code[i]}, {code[i+3], code[i+2]}, "code");
    end
    if ($urandom_range(0, 3) == 0) begin
      // Cancelling an idle context has no effect.
      @(negedge clock);
      cancel[c] = 1;
      @(negedge clock);
      cancel[c] = 0;
      evidence::check(!result_valid[c] && !running[c], "idle cancel produced a task");
    end
  end
endtask
