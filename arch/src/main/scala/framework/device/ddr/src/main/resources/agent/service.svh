task service();
  forever begin
    @(ctl.sample);
    if (!ctl.sample.reset) begin
      cycle++;
      for (int c = 0; c < 2; c++) begin
        if (req[c].sample.valid && !req[c].sample.ready) request_stalls++;
        if (resp[c].sample.valid && !resp[c].sample.ready) response_stalls++;
        if (req[c].sample.valid && req[c].sample.ready) begin
          result_item expected = new;
          req_t q = req[c].sample.bits;
          int key = (c << 12) | int'(q[`DF(REQ, ID)]);
          bit [511:0] line;
          verify_contract(!commands.exists(key), "Accepted duplicate live client tag");
          verify_contract(ddr_ref_read(model, q[`DF(REQ, ADDR)], line),
                          "Uninitialized fixture line");
          commands[key] = '{q, c, 0};
          expected.set_transaction_id(key);
          expected.client = c;
          expected.tag = q[`DF(REQ, ID)];
          expected.error = q[`DF(REQ, WRITE)] ? write_error.exists(q[`DF(REQ, ADDR)]) :
              read_error.exists(q[`DF(REQ, ADDR)]);
          expected.data = (q[`DF(REQ, WRITE)] || expected.error) ? '0 : line;
          scoreboard.expected_export.write(expected);
          accepted++;
          behavior.sample(c, q[`DF(REQ, WRITE)], q[`DF(REQ, MASK)] == 0 ? 0 : (q[`DF(REQ, MASK)
                          ] == 64'hffffffffffffffff ? 1 : 2), expected.error, commands.num(), 0);
          if (commands.num() > peak) peak = commands.num();
        end
      end
      if (aw.sample.valid && !aw.sample.ready) aw_stalls++;
      if (w.sample.valid && !w.sample.ready) w_stalls++;
      if (aw.sample.valid && aw.sample.ready) begin
        int key = observe_address(aw.sample.bits, 1);
        int id = aw.sample.bits[`DF(A, ID)];
        aw_ids.push_back(id);
        aw_keys.push_back(key);
        aw_serials.push_back(aw_count);
        aw_count++;
      end
      if (w.sample.valid && w.sample.ready) begin
        verify_contract(!$isunknown(w.sample.bits), "Unknown W payload");
        verify_contract(w.sample.bits[`DF(W, LAST)] == (assembling_beat == `DDR_BEATS - 1),
                        "Wrong WLAST");
        assembling.data[assembling_beat*`DDR_DATA_BITS+:`DDR_DATA_BITS] = w.sample.bits[
        `DF(W, DATA)
        ];
        assembling.mask[assembling_beat*(`DDR_BEAT_BYTES)+:(`DDR_BEAT_BYTES)] = w.sample.bits[
        `DF(W, STRB)
        ];
        if (assembling_beat == `DDR_BEATS - 1) begin
          w_lines.push_back(assembling);
          assembling = '0;
          assembling_beat = 0;
          w_bursts++;
        end else assembling_beat++;
      end
      if (aw_ids.size() && w_lines.size()) begin
        int   id = aw_ids.pop_front();
        int   key = aw_keys.pop_front();
        int   serial = aw_serials.pop_front();
        wline line = w_lines.pop_front();
        req_t q = commands[key].packet;
        verify_contract(!writing_ids.exists(id) && !reading_ids.exists(id),
                        "AXI ID reused while active");
        verify_contract(line.data === q[`DF(REQ, DATA)] && line.mask === q[`DF(REQ, MASK)],
                        "AW/W order, payload or byte mask mismatch");
        verify_contract(ddr_ref_write(model, q[`DF(REQ, ADDR)], line.data, line.mask),
                        "Write outside initialized DDR");
        writing_ids[id] = '{key, cycle + 7, serial};
      end
      if (ar.sample.valid && ar.sample.ready) begin
        int key = observe_address(ar.sample.bits, 0);
        int id = ar.sample.bits[`DF(A, ID)];
        bit [511:0] line;
        verify_contract(!reading_ids.exists(id) && !writing_ids.exists(id),
                        "AXI ID reused while active");
        verify_contract(ddr_ref_read(model, ar.sample.bits[`DF(A, ADDR)], line),
                        "Read outside initialized DDR");
        reading_ids[id] = '{key, 0, ar_count, line};
        ar_count++;
      end
      if (r.sample.valid && r.sample.ready) begin
        int key = reading_ids[r_id].key;
        r_beats++;
        if (last_r_id >= 0 && last_r_id != r_id && reading_ids.exists(
                last_r_id
            ) && reading_ids[last_r_id].serial == last_r_serial)
          interleaves++;
        last_r_id = r_id;
        last_r_serial = reading_ids[r_id].serial;
        if (reading_ids[r_id].beat == `DDR_BEATS - 1) begin
          if (reading_ids[r_id].serial < max_r_serial) r_reorders++;
          else max_r_serial = reading_ids[r_id].serial;
          commands[key].done = 1;
          reading_ids.delete(r_id);
        end else reading_ids[r_id].beat++;
        r_active = 0;
        r_cursor = (r_id + 1) % 16;
      end
      if (b.sample.valid && b.sample.ready) begin
        if (writing_ids[b_id].serial < max_b_serial) b_reorders++;
        else max_b_serial = writing_ids[b_id].serial;
        commands[writing_ids[b_id].key].done = 1;
        writing_ids.delete(b_id);
        b_active = 0;
      end
      for (int c = 0; c < 2; c++)
      if (resp[c].sample.valid && resp[c].sample.ready) begin
        result_item actual = new;
        int key = (c << 12) | int'(resp[c].sample.bits[`DF(RESP, ID)]);
        verify_contract(!$isunknown(resp[c].sample.bits) && commands.exists(key
                        ) && commands[key].done, "Early/unknown line response before final R or B");
        actual.set_transaction_id(key);
        actual.client = c;
        actual.tag = resp[c].sample.bits[`DF(RESP, ID)];
        actual.data = resp[c].sample.bits[`DF(RESP, DATA)];
        actual.error = resp[c].sample.bits[`DF(RESP, ERROR)];
        scoreboard.actual_export.write(actual);
        if (actual.error) error_results++;
        returned[c]++;
        commands.delete(key);
      end
    end
    @(negedge ctl.clock);
    aw.ready = !ctl.reset && !hold_aw && cycle % 5 != 0;
    w.ready  = !ctl.reset && !hold_w && cycle % 7 >= 2;
    ar.ready = !ctl.reset && !hold_ar && cycle % 4 != 0;
    for (int c = 0; c < 2; c++) resp[c].ready = !ctl.reset && !hold_resp[c] && cycle % 5 != c;
    if (!r_active && !hold_r)
      for (int off = 0; off < 16; off++) begin
        int id = (r_cursor + off) % 16;
        if (!r_active && reading_ids.exists(id)) begin
          int key = reading_ids[id].key;
          u64 address = commands[key].packet[`DF(REQ, ADDR)];
          int beat = reading_ids[id].beat;
          r_active = 1;
          r_id = id;
          r_packet = '0;
          r_packet[`DF(R, ID)] = id;
          r_packet[`DF(R, DATA)] = reading_ids[id].line[beat*`DDR_DATA_BITS+:`DDR_DATA_BITS];
          r_packet[`DF(R, LAST)] = beat == `DDR_BEATS - 1;
          if (read_error.exists(address) && beat == read_error[address])
            r_packet[`DF(R, RESP)] = beat % 2 ? 3 : 2;
        end
      end
    if (!b_active && !hold_b)
      for (int id = 15; id >= 0; id--)
      if (!b_active && writing_ids.exists(id) && writing_ids[id].due <= cycle) begin
        b_active = 1;
        b_id = id;
        b_packet = '0;
        b_packet[`DF(B, ID)] = id;
        if (write_error.exists(commands[writing_ids[id].key].packet[`DF(REQ, ADDR)]))
          b_packet[
          `DF(B, RESP)
          ] = write_error[commands[writing_ids[id].key].packet[
          `DF(REQ, ADDR)
          ]];
      end
    r.valid = !ctl.reset && r_active;
    r.bits  = r_packet;
    b.valid = !ctl.reset && b_active;
    b.bits  = b_packet;
  end
endtask
