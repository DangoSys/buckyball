// Random task lifecycles on both contexts: arbitrary descriptors and submitter contexts
// (address space, privilege, SUM/MXR, every PMP entry), staging in random order with a
// wrong fingerprint overwritten before Start, launch backpressure, cancellation of live and
// idle contexts, results with arbitrary values and cancellation flags, every query field,
// and fields the operation ignores. The monitor checks launches, cancels and availability.
task automatic control_random(int rounds);
  for (int round = 0; round < rounds; round++) begin
    int order[2] = '{0, 1};
    bit cancelled[2];
    logic [63:0] values[2] = '{{$urandom, $urandom}, {$urandom, $urandom}};
    if ($urandom_range(0, 1)) order = '{1, 0};
    // Query field 7 accepts any context index; Acquire/Release ignore context, field and data.
    transact(0, {$urandom, $urandom}, 7, {$urandom, $urandom}, 2);
    transact(5, $urandom, $urandom, {$urandom, $urandom});
    foreach (order[n]) begin
      int ctx = order[n];
      int fields[6] = '{0, 1, 2, 3, 4, 5};
      fields.shuffle();
      storage_busy = $urandom;
      if ($urandom_range(0, 1)) transact(1, ctx, 5, {$urandom, $urandom});
      foreach (fields[f])
      transact(1, ctx, fields[f],
               fields[f] == 5 ? signature[ctx] : fields[f] == 0 ? $urandom : {$urandom, $urandom});
      transact(0, ctx, 0, {$urandom, $urandom}, signature[ctx]);
      transact(0, ctx, 1, {$urandom, $urandom}, 1);
      transact(0, ctx, 2, 0, `CODE_BYTES);
      transact(0, ctx, 3, 0, `TLS_BASE);
      transact(0, ctx, 4, 0, `TLS_BYTES);
      transact(0, ctx, 5, 0, `TSS_BASE);
      transact(0, ctx, 6, 0, `TSS_BYTES);
      // Cancelling a context without a live task has no effect.
      if ($urandom_range(0, 3) == 0) transact(4, ctx, $urandom, {$urandom, $urandom});
      request_satp = {$urandom, $urandom};
      request_privilege = $urandom;
      request_sum = $urandom;
      request_mxr = $urandom;
      for (int i = 0; i < `PMP_BITS; i += 32) request_pmp[i+:32] = $urandom;
      start_ready[ctx] = $urandom_range(0, 1);
      fork
        transact(2, ctx, $urandom_range(0, 1) ? 7 : $urandom, {$urandom, $urandom});
        begin
          repeat ($urandom_range(0, 3)) @(negedge clock);
          start_ready[ctx] = 1;
        end
      join
      transact(0, ctx, 1, 0, 3);
    end
    foreach (order[n]) begin
      int ctx = order[n];
      cancelled[ctx] = $urandom;
      if ($urandom_range(0, 1))
        transact(4, ctx, $urandom_range(0, 1) ? 7 : $urandom, {$urandom, $urandom});
    end
    fork
      complete(0, values[0], cancelled[0]);
      complete(1, values[1], cancelled[1]);
    join
    storage_busy = $urandom;
    foreach (order[n]) begin
      int ctx = order[n];
      transact(0, ctx, 1, 0, cancelled[ctx] ? 13 : 5);
      transact(0, ctx, 8, 0, staged[ctx][0]);
      transact(0, ctx, 9, 0, values[ctx]);
      transact(3, ctx, $urandom_range(0, 1) ? 7 : $urandom, {$urandom, $urandom});
      transact(0, ctx, 1, 0, 1);
    end
    storage_busy = 0;
    transact(6, $urandom, $urandom, {$urandom, $urandom});
    if (in_use) `uvm_fatal("GROUP", "group still in use after release")
  end
endtask
