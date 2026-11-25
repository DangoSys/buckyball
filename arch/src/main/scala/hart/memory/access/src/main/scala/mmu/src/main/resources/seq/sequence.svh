class scenario_sequence extends uvm_sequence;
  `uvm_object_utils(scenario_sequence)
  env test_env;
  virtual mmu_if vif;
  localparam u64 MAX_PA = (64'h1 << `MMU_ADDRESS_BITS) - 1;
  localparam u64 HIGH_PPN = 64'h1 << (`MMU_ADDRESS_BITS - 12);
  function new(string name = "scenario_sequence");
    super.new(name);
  endfunction
  task operation(int kind, request_t payload = '0, int mode = 8, u64 root = 16, bit change = 0,
                 int next_mode = 0, u64 next_root = 0);
    request_sequence seq = request_sequence::type_id::create("operation");
    seq.item = request_item::type_id::create("item");
    seq.item.kind = kind;
    seq.item.payload = payload;
    seq.item.mode = mode;
    seq.item.root = root;
    seq.item.change_inputs = change;
    seq.item.next_mode = next_mode;
    seq.item.next_root = next_root;
    seq.start(test_env.seqr);
  endtask
  task flush();
    operation(1);
  endtask
  task probe(request_t r, int mode = 8, u64 root = 16);
    operation(2, r, mode, root);
  endtask
  task send(request_t r, int mode = 8, u64 root = 16, bit change_inputs = 0, int next_mode = 0,
            u64 next_root = 0);
    operation(0, r, mode, root, change_inputs, next_mode, next_root);
  endtask
  task run_case(string name, request_t r, int mode = 8, u64 root = 16, bit change_inputs = 0);
    int target = test_env.scoreboard.checked + 1;
    test_env.reference.current_case = name;
    send(r, mode, root, change_inputs);
    test_env.scoreboard.wait_checked(target);
    @(negedge vif.clock);
    evidence::check(mmu_ref_pending(test_env.reference.reference
                    ) == 0 && !test_env.reference.pending_result,
                    "Walker skipped expected PTE responses");
  endtask
  function request_t request(u64 va, int operation = 0, int privilege = 1, bit sum = 0,
                             bit mxr = 0);
    request_t r = '0;
    r[`RQ(VADDR)] = va;
    r[`RQ(WRITE)] = operation == 1;
    r[`RQ(EXECUTE)] = operation == 2;
    r[`RQ(PRIVILEGE)] = privilege;
    r[`RQ(SUM)] = sum;
    r[`RQ(MXR)] = mxr;
    return r;
  endfunction
  function void program_pte(u64 address, u64 value, bit error = 0);
    mmu_ref_program(test_env.reference.reference, address, value, error);
  endfunction
  function u64 pte(u64 address);
    longint unsigned value;
    int unsigned error;
    if (!mmu_ref_read(test_env.reference.reference, address, value, error))
      `uvm_fatal("FIXTURE", "Attempt to modify an unprogrammed PTE")
    return value;
  endfunction
  function u64 pte_address(u64 va, int level);
    return (64'h12 - level) * 4096 + ((va >> (12 + 9 * level)) & 511) * 8;
  endfunction
  task prepare(u64 va, int leaf_level, u64 flags = 64'hcf, u64 mapped = 64'h80000000);
    flush();
    mmu_ref_clear(test_env.reference.reference);
    for (int level = 2; level >= leaf_level; level--) begin
      if (level == leaf_level) program_pte(pte_address(va, level), ((mapped >> 12) << 10) | flags);
      else program_pte(pte_address(va, level), ((64'h13 - level) << 10) | 1);
    end
  endtask
  task table(u64 address, u64 mapped, u64 flags = 64'hcf, u64 root = 16);
    for (int level = 2; level >= 0; level--) begin
      u64 word_address = (root + 2 - level) * 4096 + ((address >> (12 + 9 * level)) & 511) * 8;
      program_pte(word_address,
                  level == 0 ? ((mapped >> 12) << 10) | flags : ((root + 3 - level) << 10) | 1);
    end
  endtask
  task body();
    u64 va = 64'h123456789, flags, mapped;
    int permissions[5] = '{1, 3, 4, 5, 7};
    int target, before_reads;
    test_env.reset.pulse(4);
    prepare(va, 0, 64'h43, 64'h81234000);
    target = test_env.scoreboard.checked + 2;
    test_env.reference.current_case = "back_to_back_cached_permission_recheck";
    test_env.reference.hold_access = 1;
    test_env.reference.hold_response = 1;
    send(request(va));
    fork
      send(request(va, 2));
      begin
        wait (`MO(test_env.reference.latest.outputs, ACCESS_VALID));
        repeat (8) @(negedge vif.clock);
        test_env.reference.hold_access = 0;
        wait (`MO(test_env.reference.latest.outputs, RESP_VALID));
        repeat (8) @(negedge vif.clock);
        test_env.reference.hold_response = 0;
      end
    join
    test_env.scoreboard.wait_checked(target);
    // Three page sizes and both canonical sign regions, including page edges.
    for (int level = 0; level < 3; level++) begin
      for (int position = 0; position < 3; position++) begin
        u64 size = 64'h1000 << (9 * level);
        u64 address = (va & ~(size - 1)) + (position == 0 ? 0 : position == 1 ? 123 : size - 1);
        prepare(address, level);
        run_case($sformatf("page_size_%0d_edge_%0d", level, position), request(address));
        address |= 64'hffffffc000000000;
        prepare(address, level);
        run_case($sformatf("negative_canonical_%0d_%0d", level, position), request(address));
      end
    end
    // Permissions: all valid RWX forms, U/S, U-page, SUM, MXR, load/store/fetch.
    foreach (permissions[i])
      for (int privilege = 0; privilege < 2; privilege++)
        for (int user_page = 0; user_page < 2; user_page++)
          for (int sum = 0; sum < 2; sum++)
            for (int mxr = 0; mxr < 2; mxr++)
              for (int operation = 0; operation < 3; operation++) begin
                flags = 64'hc1 | (u64'(permissions[i]) << 1) | (u64'(user_page) << 4);
                prepare(va, 0, flags, 64'h81234000);
                run_case($sformatf(
                         "perm_%0d_prv%0d_u%0d_sum%0d_mxr%0d_op%0d",
                         permissions[i],
                         privilege,
                         user_page,
                         sum,
                         mxr,
                         operation
                         ), request(va, operation, privilege, sum, mxr));
              end
    for (int level = 0; level < 3; level++)
      for (int ad = 0; ad < 4; ad++)
        for (int operation = 0; operation < 3; operation++) begin
          flags = 64'hf | (u64'(ad) << 6);
          prepare(va, level, flags);
          run_case($sformatf("svade_level%0d_ad%0d_op%0d", level, ad, operation), request(
                   va, operation));
        end
    for (int level = 0; level < 3; level++) begin
      for (int reserved_bit = 54; reserved_bit < 64; reserved_bit++) begin
        prepare(va, level);
        program_pte(pte_address(va, level), pte(pte_address(va, level)) | (64'h1 << reserved_bit));
        run_case($sformatf("reserved_leaf_l%0d_bit%0d", level, reserved_bit), request(va));
      end
      for (int encoding = 0; encoding < 3; encoding++) begin
        prepare(va, level, encoding == 0 ? 64'hce : encoding == 1 ? 64'hc5 : 64'hcd);
        run_case($sformatf("invalid_pte_l%0d_encoding%0d", level, encoding), request(va));
      end
      prepare(va, level);
      program_pte(pte_address(va, level), '1, 1);
      run_case($sformatf("backend_error_l%0d", level), request(va));
      prepare(va, level, 64'hcf, HIGH_PPN * 4096);
      run_case($sformatf("physical_leaf_overflow_l%0d", level), request(va));
      prepare(va, level, 64'hf, HIGH_PPN * 4096);
      run_case($sformatf("pte_fault_precedes_final_pa_overflow_l%0d", level), request(va));
    end
    for (int level = 1; level <= 2; level++) begin
      for (int bit_index = 0; bit_index < 9 * level; bit_index++) begin
        prepare(va, level);
        program_pte(pte_address(va, level), pte(pte_address(va, level)
                    ) | (64'h1 << (10 + bit_index)));
        run_case($sformatf("superpage_alignment_l%0d_bit%0d", level, bit_index), request(va));
      end
      for (int bad = 0; bad < 3; bad++) begin
        prepare(va, 0);
        program_pte(pte_address(va, level), pte(pte_address(va, level)
                    ) | (64'h1 << (bad == 0 ? 4 : bad == 1 ? 6 : 7)));
        run_case($sformatf("nonleaf_reserved_l%0d_%0d", level, bad), request(va));
      end
      for (int bit_index = 54; bit_index < 64; bit_index++) begin
        prepare(va, 0);
        program_pte(pte_address(va, level), pte(pte_address(va, level)) | (64'h1 << bit_index));
        run_case($sformatf("nonleaf_upper_reserved_l%0d_%0d", level, bit_index), request(va));
      end
      prepare(va, 0);
      program_pte(pte_address(va, level), (HIGH_PPN << 10) | 1);
      run_case($sformatf("next_table_address_overflow_l%0d", level), request(va));
    end
    prepare(va, 0);
    program_pte(pte_address(va, 0), (64'h13 << 10) | 1);
    run_case("nonleaf_at_bottom", request(va));
    prepare(va, 0, 64'h3ef);  // G and both RSW bits are permitted and ignored for translation.
    for (int level = 1; level <= 2; level++)
      program_pte(pte_address(va, level), pte(pte_address(va, level)) | 64'h320);
    run_case("software_and_global_bits", request(va));
    prepare(va, 0, 64'h59);
    run_case("latched_root_sum_mxr_privilege", request(va, 0, 1, 1, 1), 8, 16, 1);
    mmu_ref_clear(test_env.reference.reference);
    run_case("root_physical_overflow", request(va), 8, HIGH_PPN);
    run_case("positive_noncanonical", request(64'h0000004000000000));
    run_case("upper_noncanonical", request(64'hffffffbfffffffff));
    for (int operation = 0; operation < 3; operation++) begin
      for (int privilege = 0; privilege < 2; privilege++) begin
        run_case("bare_noncanonical_physical", request(64'h8000000000, operation, privilege), 0, 0);
        run_case("bare_last_physical_byte", request(MAX_PA, operation, privilege), 0, 0);
        run_case("bare_physical_overflow", request(MAX_PA + 1, operation, privilege), 0, 0);
      end
      run_case("machine_ignores_sv39", request(64'h8000000000, operation, 3), 8, HIGH_PPN);
      run_case("machine_physical_overflow", request(MAX_PA + 1, operation, 3), 8, 16);
    end
    // Single-address translations on opposite sides of a base-page boundary.
    prepare(64'hfff, 0, 64'hcf, MAX_PA & ~64'hfff);
    run_case("last_byte_before_physical_limit", request(64'hfff));
    prepare(64'h1000, 0, 64'hcf, MAX_PA + 1);
    run_case("next_page_physical_overflow", request(64'h1000));
    // The last PTE word of the highest implemented physical table page is legal.
    mmu_ref_clear(test_env.reference.reference);
    program_pte(MAX_PA - 7, 64'hcf);
    run_case("last_physical_pte_word", request(64'hffffffffc0000000), 8, MAX_PA >> 12);
    // Warm translations re-check permissions without re-reading page tables.
    prepare(va, 0, 64'h4f, 64'h81234000);
    run_case("cached_read", request(va));
    before_reads = test_env.reference.reads;
    run_case("cached_store_dirty_fault", request(va, 1));
    run_case("cached_user_fault", request(va, 0, 0));
    evidence::check(test_env.reference.reads == before_reads,
                    "Walker did not reuse its cached leaf");
    probe(request(va));
    probe(request(va, 0, 0));
    // Updating memory without SFENCE does not change an already cached leaf.
    program_pte(pte_address(va, 0), ((64'h84567000 >> 12) << 10) | 64'hcf);
    run_case("cached_old_pte", request(va));
    flush();
    run_case("after_sfence_new_pte", request(va));
    // Every configured entry and round-robin eviction, checked through the public lookup port.
    flush();
    mmu_ref_clear(test_env.reference.reference);
    for (int index = 0; index <= `MMU_TLB_ENTRIES; index++)
      table('h400000 + index * 4096, 'h90000000 + index * 4096);
    for (int index = 0; index < `MMU_TLB_ENTRIES; index++)
      run_case("fill_tlb", request('h400000 + index * 4096));
    for (int index = 0; index < `MMU_TLB_ENTRIES; index++) probe(request('h400000 + index * 4096));
    run_case("evict_oldest_tlb_entry", request('h400000 + `MMU_TLB_ENTRIES * 4096));
    for (int index = 0; index <= `MMU_TLB_ENTRIES; index++) probe(request('h400000 + index * 4096));
    // A changed live config cannot relabel a walk that was accepted under another root.
    prepare(va, 0, 64'hcf, 64'h81234000);
    target = test_env.scoreboard.checked + 1;
    test_env.reference.current_case = "latched_tlb_fill_root";
    send(request(va), 8, 16, 1, 8, 32);
    test_env.scoreboard.wait_checked(target);
    probe(request(va), 8, 16);
    probe(request(va), 8, 32);
    table(va, 64'h84567000, 64'hcf, 32);
    run_case("different_root_same_vpn", request(va), 8, 32);
    probe(request(va), 8, 16);
    probe(request(va), 8, 32);
    // SFENCE during an outstanding PTE response must prevent a later stale TLB refill.
    prepare(va, 0, 64'hcf, 64'h81234000);
    target = test_env.scoreboard.checked + 1;
    test_env.reference.current_case = "flush_during_walk";
    test_env.reference.hold_result = 1;
    send(request(va));
    wait (test_env.reference.pending_result);
    flush();
    test_env.reference.hold_result = 0;
    test_env.scoreboard.wait_checked(target);
    probe(request(va));
    // SFENCE wins when the final leaf response and flush share a clock edge.
    prepare(va, 0, 64'hcf, 64'h81234000);
    target = test_env.scoreboard.checked + 1;
    test_env.reference.current_case = "flush_on_fill";
    send(request(va));
    wait (test_env.reference.pending_result && mmu_ref_pending(test_env.reference.reference) == 0);
    test_env.reference.hold_result = 1;
    repeat (5) @(negedge vif.clock);
    fork
      flush();
      begin
        wait (`MI(vif.inputs, FLUSH));
        test_env.reference.hold_result = 0;
      end
    join
    test_env.scoreboard.wait_checked(target);
    probe(request(va));
    // Reset cancellation at the offer, memory wait and held translation response boundaries.
    for (int phase_index = 0; phase_index < 3; phase_index++) begin
      prepare(va, 0, 64'hcf, 64'h81234000);
      test_env.reference.current_case  = "reset_cancel_translation";
      test_env.reference.hold_access   = phase_index == 0;
      test_env.reference.hold_result   = phase_index == 1;
      test_env.reference.hold_response = phase_index == 2;
      send(request(va));
      if (phase_index == 0) wait (`MO(test_env.reference.latest.outputs, ACCESS_VALID));
      else if (phase_index == 1) wait (test_env.reference.pending_result);
      else wait (`MO(test_env.reference.latest.outputs, RESP_VALID));
      test_env.reset.pulse(3);
      test_env.reference.hold_access   = 0;
      test_env.reference.hold_result   = 0;
      test_env.reference.hold_response = 0;
      run_case("after_reset_rewalk", request(va));
    end
    evidence::check(evidence::cancelled == 3, "Walker reset cancellation count mismatch");
    `uvm_info("WALKER", $sformatf("accepted=%0d returned=%0d PTEreads=%0d reset_cancelled=%0d",
                                  test_env.reference.accepted, test_env.reference.returned,
                                  test_env.reference.reads, test_env.reference.cancelled), UVM_LOW)
  endtask
endclass
class invalid_sequence extends uvm_sequence #(request_item);
  `uvm_object_utils(invalid_sequence)
  function new(string name = "invalid_sequence");
    super.new(name);
  endfunction
  task body();
    request_item req = request_item::type_id::create("invalid");
    string scenario;
    if (!$value$plusargs("CASE=%s", scenario))
      `uvm_fatal("CASE", "Walker assertion scenario is required")
    start_item(req);
    req.payload = '0;
    req.mode = 8;
    req.root = 16;
    req.payload[`RQ(PRIVILEGE)] = 1;
    case (scenario)
      "mode": req.mode = 9;
      "privilege": req.payload[`RQ(PRIVILEGE)] = 2;
      "access": begin
        req.payload[`RQ(WRITE)]   = 1;
        req.payload[`RQ(EXECUTE)] = 1;
      end
      default: `uvm_fatal("CASE", "Unknown walker assertion scenario")
    endcase
    finish_item(req);
    #100ns;
    `uvm_fatal("EXPECTED_ASSERTION", "Invalid walker input did not trigger its assertion")
  endtask
endclass
