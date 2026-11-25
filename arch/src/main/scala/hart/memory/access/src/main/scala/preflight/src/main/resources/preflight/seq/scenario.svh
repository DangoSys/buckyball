task submit(command_t q);
  @(negedge ctl.clock);
  cmd.bits  = q;
  cmd.valid = 1;
  do @(cmd.sample); while (!cmd.sample.ready);
  @(negedge ctl.clock);
  cmd.valid = 0;
  cmd.bits  = ~q;
endtask
task drain();
  wait (live.num() == 0);
  @(ctl.sample);
  verify_contract(!auth_pending && !pte_pending && !pte_grant,
                  "Preparation endpoints did not drain");
endtask
task execute();
  u64 root, root2;
  int prior, gap_stride;
  command_t q;
  ctl.reset = 1;
  cmd.valid = 0;
  cmd.bits = '0;
  prepared.ready = 0;
  authorization.ready = 0;
  permission.valid = 0;
  permission.bits = '0;
  pte_req.ready = 0;
  pte_resp.valid = 0;
  pte_resp.bits = '0;
  repeat (5) @(negedge ctl.clock);
  ctl.reset = 0;
  fork
    service();
  join_none
  // Current BERT maximum: 1024 dense 16B rows normalize to five pages.
  root = mapping('h400000f0, 9);
  root2 = mapping('h50000000, 9);
  hold_output = 1;
  submit(request(1, 'h400000f0, 1024, 16, 16, 1, 0, 0, 8, root));
  submit(request(2, 'h50000000, 2048, 16, 16, 1, 0, 1, 8, root2));  // exactly eight pages
  submit(request(3, 'h60000ff8, 1, 16, 16));  // alignment-expanded beat crosses page
  submit(request(4, 'h70000000, 1, 16, 16));
  wait (prepared.sample.valid);
  repeat (8) @(ctl.sample);
  fork
    submit(request(5, 'h70002000, 1, 16, 16));
    begin
      repeat (12) @(ctl.sample);
      hold_output = 0;
    end
  join
  drain();
  submit(request(6, 'h50000000, 2304, 16, 16, 1, 0, 0, 8, root2));
  drain();  // ninth page: one error, zero successful output
  submit(request(7, 'h60000000, 2, 16, 8192, 2, 4096));
  drain();  // sparse 2D, four extents
  submit(request(8, 'h60000000, 1, 16, 16));
  drain();  // progress immediately after capacity error
  root = mapping('h40010000, 1, 'h43);
  submit(request(9, 'h40010000, 1, 16, 16, 1, 0, 1, 8, root));
  drain();
  root = mapping('h40020000, 1, 'h7);
  submit(request(10, 'h40020000, 1, 16, 16, 1, 0, 0, 8, root));
  drain();
  root = mapping('h40030000, 1, 'hc7, 0);
  submit(request(11, 'h40030000, 1, 16, 16, 1, 0, 0, 8, root));
  drain();
  root = mapping('h40040000, 1);
  preflight_ref_policy(model, 1, (root << 12) + ((64'h40040000 >> 30) & 511) * 8, 0, 0, 0);
  submit(request(12, 'h40040000, 1, 16, 16, 1, 0, 0, 8, root));
  drain();
  preflight_ref_policy(model, 0, 0, 1, 'h800000, 'h80000f);
  submit(request(13, 'h40040000, 1, 16, 16, 1, 0, 0, 8, root));
  drain();
  preflight_ref_policy(model, 0, 0, 1, 'h71000000, 'h71000003);
  // Denied bytes precede the user span: the aligned transport range is still authorized.
  submit(request(14, 'h71000004, 1, 4, 4));
  drain();
  preflight_ref_policy(model, 0, 0, 0, 0, 0);
  submit(request(15, (64'h1 << `PF_ADDRESS_BITS) - 16, 1, 32, 32));
  drain();
  submit(request(16, 64'hfffffffffffffff8, 1, 16, 16));
  drain();
  submit(request(17, 'h80000000, 0, 16, 16));
  drain();
  q = request(18, 'h80000000, 1, 16, 16);
  q[`PF(CMD, MODE)] = 9;
  submit(q);
  drain();
  root = mapping('h40050000, 1, 'h59);
  submit(request(19, 'h40050000, 1, 16, 16, 1, 0, 0, 8, root, 1, 1, 1));
  drain();
  // External snapshot service captures policy at command acceptance, not from later live settings.
  hold_permission = 1;
  submit(request(20, 'h72000000, 1, 16, 16));
  preflight_ref_policy(model, 0, 0, 1, 'h72000000, 'h7200000f);
  wait (auth_pending);
  repeat (8) @(ctl.sample);
  hold_permission = 0;
  drain();
  preflight_ref_policy(model, 0, 0, 0, 0, 0);
  submit(request(21, 'h73000000, 1, 16, 256, 16, 16));
  drain();  // columns retains all 32 input bits
  q = request(22, 64'hfffffffffffff000, -1, -1, -1, -1, -1);
  submit(q);
  drain();
  // Write authorization covers actual bytes; read authorization covers transport padding.
  prior = failures[4];
  preflight_ref_policy(model, 0, 0, 1, 'h74000000, 'h74000003);
  submit(request(23, 'h74000004, 1, 4, 4, 1, 0, 1));
  drain();
  verify_contract(failures[4] == prior, "write was denied for untouched prefix padding");
  submit(request(24, 'h74000004, 1, 4, 4));
  drain();
  verify_contract(failures[4] == prior + 1, "read failed to authorize actual prefix overread");
  prior = failures[4];
  preflight_ref_policy(model, 0, 0, 1, 'h74000108, 'h7400013f);
  submit(request(25, 'h74000104, 1, 4, 4, 1, 0, 1));
  drain();
  verify_contract(failures[4] == prior, "write was denied for untouched tail padding");
  submit(request(26, 'h74000104, 1, 4, 4));
  drain();
  verify_contract(failures[4] == prior + 1, "read failed to authorize actual tail overread");
  prior = failures[4];
  preflight_ref_policy(model, 0, 0, 1, 'h74002ff0, 'h74002ffb);
  submit(request(27, 'h74002ffc, 1, 8, 8, 1, 0, 1));
  drain();
  verify_contract(failures[4] == prior, "cross-page write included untouched bytes");
  submit(request(28, 'h74002ffc, 1, 8, 8));
  drain();
  verify_contract(failures[4] == prior + 1, "cross-page read omitted padded transport bytes");
  preflight_ref_policy(model, 0, 0, 0, 0, 0);
  // MobileNet's gather has nine aligned segments on two noncontiguous PA pages.
  // Every same-page gap must be explicitly authorized before the plan retains a union.
  root  = mapping('h74003e00, 2);
  prior = failures[5];
  submit(request(29, 'h74003e00, 1, 16, 4032, 8, 72, 0, 8, root));
  drain();
  verify_contract(failures[5] == prior, "Authorized MobileNet gather exceeded range capacity");
  // Splitting preserves the exact accessed bytes and keeps each list within eight records.
  prior = failures[5];
  submit(request(30, 'h74003e00, 1, 16, 4032, 4, 72, 0, 8, root));
  drain();
  submit(request(31, 'h74003f20, 1, 16, 4032, 4, 72, 0, 8, root));
  drain();
  verify_contract(failures[5] == prior, "Split MobileNet gather exceeded range capacity");
  // A denied hole must keep both individually legal spans, rather than poison the command.
  gap_stride = `PF_BEAT_BYTES == 16 ? 72 : 2 * `PF_BEAT_BYTES + 8;
  preflight_ref_policy(model, 0, 0, 1, 'h800e00 + `PF_BEAT_BYTES,
                       'h800e00 + (gap_stride / `PF_BEAT_BYTES) * `PF_BEAT_BYTES - 1);
  prior = failures[4];
  submit(request(32, 'h74003e00, 1, 16, 4032, 2, gap_stride, 0, 8, root));
  drain();
  verify_contract(failures[4] == prior, "Denied union rejected individually authorized pixels");
  // Denial of an actual transport byte must still yield one terminal fault, no partial plan.
  preflight_ref_policy(model, 0, 0, 1, 'h800e00 + gap_stride, 'h800e00 + gap_stride + 7);
  submit(request(33, 'h74003e00, 1, 16, 4032, 2, gap_stride, 0, 8, root));
  drain();
  verify_contract(failures[4] == prior + 1, "Actual MobileNet pixel bytes escaped authorization");
  preflight_ref_policy(model, 0, 0, 0, 0, 0);
  verify_contract(
      peak==4&&submitted==retired&&command_stalls>0&&output_stalls>0&&auth_stalls>0,
      "Missing concurrency/backpressure/drain coverage");
  for (int error = 1; error < 7; error++)
    verify_contract(failures[error] > 0, "Missing typed error scenario");
  verify_contract(scoreboard.checked == segments, "Not every output descriptor was compared");
  `uvm_info(
      "PREFLIGHT_PASS",
      $sformatf(
          "commands=%0d descriptors=%0d peak=%0d PTE=%0d auth=%0d errors shape/overflow/PF/AF/capacity/context=%0d/%0d/%0d/%0d/%0d/%0d stalls cmd/out/auth=%0d/%0d/%0d; all plans/permissions drained",
          retired, segments, peak, pte_count, auth_count, failures[1], failures[2], failures[3],
          failures[4], failures[5], failures[6], command_stalls, output_stalls, auth_stalls),
      UVM_LOW)
endtask
