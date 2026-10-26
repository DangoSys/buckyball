task service();
  forever begin
    @(ctl.sample);
    if (!ctl.sample.reset) begin
      if (authorization.sample.valid)
        behavior.sample(0, authorization.sample.bits[`PF(AUTH, WRITE)], 0,
                        authorization.sample.bits[`PF(AUTH, ISPTE)], !authorization.sample.ready);
      if (prepared.sample.valid)
        behavior.sample(prepared.sample.bits[`PF(OUT, ERROR)], prepared.sample.bits[`PF(OUT, WRITE)
                        ], prepared.sample.bits[`PF(OUT, LAST)], 0, !prepared.sample.ready);
      cycle++;
      if (cmd.sample.valid && !cmd.sample.ready) command_stalls++;
      if (prepared.sample.valid && !prepared.sample.ready) output_stalls++;
      if (authorization.sample.valid && !authorization.sample.ready) auth_stalls++;
      if (cmd.sample.valid && cmd.sample.ready) predict(cmd.sample.bits);
      if (permission.sample.valid && permission.sample.ready) begin
        verify_contract(auth_pending, "Unsolicited permission consumed");
        if (auth_pte && auth_allow) begin
          pte_grant = 1;
          grant_id  = auth_id;
        end
        auth_pending = 0;
      end
      if (authorization.sample.valid && authorization.sample.ready) begin
        int unsigned allow;
        int id = authorization.sample.bits[`PF(AUTH, ID)];
        verify_contract(!$isunknown(authorization.sample.bits) && !auth_pending,
                        "Overlapping/unknown authorization");
        verify_contract(preflight_ref_authorize(
                        model,
                        id,
                        authorization.sample.bits[
                        `PF(AUTH, PA)
                        ],
                        authorization.sample.bits[
                        `PF(AUTH, BYTES)
                        ],
                        authorization.sample.bits[
                        `PF(AUTH, WRITE)
                        ],
                        authorization.sample.bits[
                        `PF(AUTH, ISPTE)
                        ],
                        authorization.sample.bits[
                        `PF(AUTH, PRIVILEGE)
                        ],
                        allow
                        ) == 1, "Authorization lost snapshot, range or PTE S privilege");
        auth_pending = 1;
        auth_id = id;
        auth_allow = allow;
        auth_pte = authorization.sample.bits[`PF(AUTH, ISPTE)];
        grant_pa = authorization.sample.bits[`PF(AUTH, PA)];
        auth_due = cycle + 3;
        permit_packet = '0;
        permit_packet[`PF(PERMIT, ID)] = id;
        permit_packet[`PF(PERMIT, ALLOW)] = allow;
        auth_count++;
      end
      if (pte_resp.sample.valid && pte_resp.sample.ready) pte_pending = 0;
      if (pte_req.sample.valid && pte_req.sample.ready) begin
        u64 value;
        int unsigned error;
        verify_contract(pte_grant && !pte_pending && !$isunknown(pte_req.sample.bits),
                        "PTE accessed before authorization");
        verify_contract(pte_req.sample.bits[`PF(PTE, ADDR)] == grant_pa && grant_pa[2:0] == 0,
                        "PTE request changed address or is not 8-byte aligned");
        verify_contract(preflight_ref_pte(model, grant_id, grant_pa, value, error) == 1,
                        "PTE address chain differs from architectural walker");
        pte_packet = '0;
        pte_packet[`PF(PTERESP, DATA)] = value;
        pte_packet[`PF(PTERESP, ERROR)] = error;
        pte_id = grant_id;
        pte_pending = 1;
        pte_due = cycle + 2 + (pte_count % 4);
        pte_grant = 0;
        pte_count++;
      end
      if (prepared.sample.valid) begin
        int id = prepared.sample.bits[`PF(OUT, ID)];
        verify_contract(live.exists(id) && !$isunknown(prepared.sample.bits),
                        "Unsolicited/unknown prepared plan");
        verify_contract(
            preflight_ref_pending(model, id
            ) == 0 && !(auth_pending && auth_id == id) && !(pte_pending && pte_id == id),
            "Partial plan exposed before all translations/permissions completed");
        if (prepared.sample.ready) begin
          result_item actual = new;
          actual.packet = prepared.sample.bits;
          actual.set_transaction_id((id << 4) | output_index[id]);
          scoreboard.actual_export.write(actual);
          output_index[id]++;
          segments++;
          if (prepared.sample.bits[`PF(OUT, LAST)]) begin
            failures[int'(prepared.sample.bits[`PF(OUT, ERROR)])]++;
            retired++;
            live.delete(id);
            output_index.delete(id);
            preflight_ref_retire(model, id);
          end
        end
      end
    end
    @(negedge ctl.clock);
    authorization.ready = !ctl.reset && !auth_pending && cycle % 5 >= 2;
    permission.valid = !ctl.reset && auth_pending && !hold_permission && cycle >= auth_due;
    permission.bits = permit_packet;
    pte_req.ready = !ctl.reset && !pte_pending && cycle % 4 != 0;
    pte_resp.valid = !ctl.reset && pte_pending && cycle >= pte_due;
    pte_resp.bits = pte_packet;
    prepared.ready = !ctl.reset && !hold_output && cycle % 7 >= 3;
  end
endtask
