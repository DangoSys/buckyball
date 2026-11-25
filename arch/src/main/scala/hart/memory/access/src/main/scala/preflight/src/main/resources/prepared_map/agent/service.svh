task monitor();
  forever begin
    @(ctl.sample);
    if (ctl.sample.reset) begin
      pmap_ref_reset(model);
      recent[0].delete();
      recent[1].delete();
      notified.delete();
      status.delete();
    end else begin
      query_check(0, ctl.sample.query_0, ctl.sample.result_0);
      query_check(1, ctl.sample.query_1, ctl.sample.result_1);
      if (ctl.sample.query_0[`PF(QUERY, VALID)] && ctl.sample.query_1[`PF(QUERY, VALID)]) dual++;
      if (mapped.sample.valid) begin
        int error, id;
        u64 va;
        id = mapped.sample.bits[`PF(MAP_READY, ID)];
        verify_contract(pmap_ref_ready(model, id, mapped.sample.ready, error, va) == 1,
                        "unexpected notification");
        verify_contract(mapped.sample.bits[`PF(MAP_READY, ERROR)
                        ] === 3'(error) && mapped.sample.bits[`PF(MAP_READY, VA)] === va,
                        "notification mismatch");
        behavior.sample(0, 0, 1, error);
        if (mapped.sample.ready) begin
          notified[id] = 1;
          status[id]   = error;
          notifications++;
        end else stalls++;
      end
      if (retire.sample.valid && retire.sample.ready) begin
        int id;
        id = retire.sample.bits[`PF(RELEASE, ID)];
        pmap_ref_release(model, id);
        notified.delete(id);
        retired++;
      end
      if (reserve.sample.valid && reserve.sample.ready) begin
        pmap_ref_reserve(model, reserve.sample.bits[`PF(RESERVE, ID)]);
        allocations++;
      end
      if (prepared.sample.valid && prepared.sample.ready)
        pmap_ref_prepared(model, prepared.sample.bits[`PF(OUT, ID)], prepared.sample.bits[
                          `PF(OUT, VA)], prepared.sample.bits[`PF(OUT, PA)], prepared.sample.bits[
                          `PF(OUT, BYTES)], prepared.sample.bits[`PF(OUT, WRITE)],
                          prepared.sample.bits[`PF(OUT, LAST)], prepared.sample.bits[`PF(OUT, ERROR)
                          ]);
    end
  end
endtask
