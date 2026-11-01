// Random RV64IM programs on both contexts through the management service, in lockstep
// with the reference hart. TSS writes, private-store transfers (both directions, code and
// data), descriptor staging, Start with a random submitter context, cancellation, result
// queries and acknowledgement all go through management requests serialized between the
// two context threads. Completed tasks have their TLS window, TSS window and code read
// back through the TSS.
task automatic tss_write(longint unsigned offset, longint unsigned value);
  command(12, 0, offset, value);
endtask

task automatic tss_check(longint unsigned offset, longint unsigned expected, string what);
  longint unsigned reply;
  request(11, 0, offset, 0, reply);
  evidence::check(reply === expected, $sformatf(
                  "%s at TSS+%h: %h expected %h", what, offset, reply, expected));
endtask

// op 7/8: code to/from TSS; op 9/10: TLS to/from TSS.
task automatic transfer(int ctx, bit code, bit to_local, int local_offset, int shared_offset,
                        int bytes);
  command((code ? 7 : 9) + to_local, ctx, local_offset, (longint'(bytes) << 32) | shared_offset);
endtask

// Every query field of an idle context; the context count accepts any context index.
task automatic queries(int c);
  longint unsigned expected[8] = '{
      c ? `SIGNATURE_1 : `SIGNATURE_0,
      1,
      `CODE_BYTES,
      `TLS_BASE,
      `TLS_BYTES,
      `TSS_BASE,
      `TSS_BYTES,
      2
  };
  foreach (expected[f]) begin
    longint unsigned reply;
    request(0, f == 7 ? $urandom : c, f, {$urandom, $urandom}, reply);
    evidence::check(reply === expected[f], $sformatf(
                    "context %0d query %0d: %h expected %h", c, f, reply, expected[f]));
  end
endtask

task automatic programs(int c, int count);
  for (int n = 0; n < count; n++) begin
    int unsigned code[$];
    longint unsigned tls_window = `TLS_BASE + $urandom_range(0, 15) * 1024;
    longint unsigned tss_window = `TSS_BASE + c * 4096 + $urandom_range(0, 3) * 1024;
    int staging = 'h2000 + c * 'h1000;
    int origin, bytes, lead = $urandom_range(0, 1), words;
    int unsigned task_id = $urandom;
    longint unsigned word, expected, status;
    longint unsigned entry, code_end, argument, stack;
    bit cancelled = 0;
    queries(c);
    // Cancelling a context without a live task has no effect.
    if ($urandom_range(0, 3) == 0) command(4, c, $urandom, {$urandom, $urandom});
    generate_program($urandom_range(10, 150), tls_window, tss_window, code);
    // An optional word before the entry makes entry and codeEnd only word aligned.
    if (lead) code.push_front($urandom);
    words  = code.size();
    origin = $urandom_range(0, (`CODE_BYTES - 4 * code.size() - 16) / 8) * 8;
    bytes  = (4 * code.size() + 7) & ~7;
    while (4 * code.size() < bytes) code.push_back($urandom);
    for (int offset = 0; offset < 1024; offset += 8) begin
      word = {$urandom, $urandom};
      tss_write(staging + offset, word);
      ant_hart_store(harts[c], tls_window + offset, word);
      word = {$urandom, $urandom};
      tss_write(tss_window - `TSS_BASE + offset, word);
      ant_hart_store(harts[c], tss_window + offset, word);
    end
    transfer(c, 0, 1, tls_window - `TLS_BASE, staging, 1024);
    for (int i = 0; i < code.size(); i += 2) begin
      tss_write(staging + 4 * i, {code[i+1], code[i]});
      ant_hart_code(harts[c], origin + 4 * i, code[i]);
      ant_hart_code(harts[c], origin + 4 * i + 4, code[i+1]);
    end
    transfer(c, 1, 1, origin, staging, bytes);
    entry = origin + 4 * lead;
    code_end = origin + 4 * words;
    argument = `TLS_BASE + $urandom_range(0, `TLS_BYTES - 1);
    stack = $urandom_range(0, 3) ? `TLS_BASE + 16 * $urandom_range(1, `TLS_BYTES / 16) : `TLS_END;
    // A wrong fingerprint may be staged first; the correct one overwrites it.
    if ($urandom_range(0, 1)) command(1, c, 5, {$urandom, $urandom});
    command(1, c, 0, task_id);
    command(1, c, 1, entry);
    command(1, c, 2, code_end);
    command(1, c, 3, argument);
    command(1, c, 4, stack);
    command(1, c, 5, c ? `SIGNATURE_1 : `SIGNATURE_0);
    ant_hart_start(harts[c], entry, argument, stack);
    chaos[c] = 1;
    checking[c] = 1;
    command(2, c, $urandom, {$urandom, $urandom});
    if ($urandom_range(0, 4) == 0) begin
      repeat ($urandom_range(0, 300)) @(negedge clock);
      // The Ant may finish before the cancel; it is cancelled until its result reaches the
      // controller. No instruction retires after an accepted cancel.
      request(4, c, $urandom, {$urandom, $urandom}, word);
      checking[c] = 0;
      cancelled   = 1;
    end
    do request(0, c, 1, 0, status); while (!(status & 4));
    chaos[c] = 0;
    drained[c] = 1;
    command_ready[c] = 1;
    if (status == 5) begin
      longint unsigned pc, data;
      int unsigned instruction, rd;
      checking[c] = 0;
      evidence::check(ant_hart_step(harts[c], 0, pc, instruction, rd, data) == 1, $sformatf(
                      "context %0d: task ended away from its ECALL at %h", c, pc));
      expected  = data;
      cancelled = 0;
    end else begin
      evidence::check(cancelled && status == 13, $sformatf("context %0d: status %0d", c, status));
      expected = 0;
    end
    finish(c, task_id, expected, cancelled);
    if (!cancelled) begin
      for (int offset = 0; offset < 1024; offset += 8)
      tss_check(tss_window - `TSS_BASE + offset, ant_hart_load(harts[c], tss_window + offset),
                "TSS window");
      transfer(c, 0, 0, tls_window - `TLS_BASE, staging, 1024);
      for (int offset = 0; offset < 1024; offset += 8)
      tss_check(staging + offset, ant_hart_load(harts[c], tls_window + offset), "TLS window");
      transfer(c, 1, 0, origin, staging, bytes);
      for (int i = 0; i < code.size(); i += 2)
      tss_check(staging + 4 * i, {code[i+1], code[i]}, "code");
    end
  end
endtask
