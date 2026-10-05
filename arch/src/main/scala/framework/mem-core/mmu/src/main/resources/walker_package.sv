`include "walker_config.svh"
`define WF(K, F) `WALK_``K``_``F``_OFFSET +: `WALK_``K``_``F``_WIDTH
package walker_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"

  import "DPI-C" function chandle mmu_ref_create(input int unsigned address_bits);
  import "DPI-C" function void mmu_ref_destroy(input chandle model);
  import "DPI-C" function void mmu_ref_clear(input chandle model);
  import "DPI-C" function void mmu_ref_program(
    input chandle model,
    input longint unsigned address,
    value,
    input int unsigned error
  );
  import "DPI-C" function int unsigned mmu_ref_read(
    input chandle model,
    input longint unsigned address,
    output longint unsigned value,
    output int unsigned error
  );
  import "DPI-C" function int unsigned mmu_ref_pending(input chandle model);
  import "DPI-C" function int unsigned mmu_ref_peek(
    input chandle model,
    output longint unsigned address
  );
  import "DPI-C" function int unsigned mmu_ref_consume(
    input chandle model,
    input longint unsigned address
  );
  import "DPI-C" function int unsigned mmu_ref_translate(
    input chandle model,
    input longint unsigned address,
    input int unsigned mode,
    input longint unsigned root,
    input int unsigned privilege,
    write,
    execute,
    sum,
    mxr,
    output longint unsigned paddr,
    output int unsigned page_fault,
    access_fault,
    level
  );
  typedef bit [63:0] u64;
  typedef logic [`WALK_REQ_WIDTH-1:0] req_t;
  class translation extends uvm_sequence_item;
    bit [63:0] paddr;
    bit page_fault, access_fault;
    bit [1:0] level;
    `uvm_object_utils_begin(translation)
      `uvm_field_int(paddr, UVM_DEFAULT)
      `uvm_field_int(page_fault, UVM_DEFAULT)
      `uvm_field_int(access_fault, UVM_DEFAULT)
      `uvm_field_int(level, UVM_DEFAULT)
    `uvm_object_utils_end
    function new(string name = "translation");
      super.new(name);
    endfunction
  endclass
  class protocol_test extends ip_test;
    `uvm_component_utils(protocol_test)
    virtual walker_control_if control;
    virtual stream_if #(`WALK_REQ_WIDTH) req;
    virtual stream_if #(`WALK_RESP_WIDTH) resp;
    virtual stream_if #(`WALK_ACCESS_WIDTH) access;
    virtual stream_if #(`WALK_RESULT_WIDTH) result;
    in_order_scoreboard #(translation) scoreboard;
    // Only explicitly programmed PTEs exist. There is no implicit zero/pattern RAM.
    chandle model;
    int
        expected = 0,
        accepted = 0,
        returned = 0,
        reads = 0,
        cycle = 0,
        req_stalls = 0,
        access_stalls = 0,
        resp_stalls = 0;
    int page_faults = 0, access_faults = 0, successes[3] = '{0, 0, 0};
    int due = 0;
    bit pending_result = 0, result_active = 0, hold_access = 0, hold_response = 0;
    logic [`WALK_RESULT_WIDTH-1:0] result_packet;
    string current_case;
    localparam u64 MAX_PA = (64'h1 << `WALK_RESP_PADDR_WIDTH) - 1;
    localparam u64 HIGH_PPN = 64'h1 << (`WALK_RESP_PADDR_WIDTH - 12);
    function new(string name, uvm_component parent);
      super.new(name, parent);
      timeout = 1ms;
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      scoreboard = in_order_scoreboard#(translation)::type_id::create("scoreboard", this);
      model = mmu_ref_create(`WALK_RESP_PADDR_WIDTH);
      if (!uvm_config_db#(virtual walker_control_if)::get(
              this, "", "control", control
          ) || !uvm_config_db#(virtual stream_if #(`WALK_REQ_WIDTH))::get(
              this, "", "req", req
          ) || !uvm_config_db#(virtual stream_if #(`WALK_RESP_WIDTH))::get(
              this, "", "resp", resp
          ) || !uvm_config_db#(virtual stream_if #(`WALK_ACCESS_WIDTH))::get(
              this, "", "access", access
          ) || !uvm_config_db#(virtual stream_if #(`WALK_RESULT_WIDTH))::get(
              this, "", "result", result
          ))
        `uvm_fatal("VIF", "Walker interfaces missing")
    endfunction
    function req_t request(u64 va, int operation = 0, int privilege = 1, bit sum = 0, bit mxr = 0);
      req_t r = '0;
      r[`WF(REQ, VADDR)] = va;
      r[`WF(REQ, WRITE)] = operation == 1;
      r[`WF(REQ, EXECUTE)] = operation == 2;
      r[`WF(REQ, PRIVILEGE)] = privilege;
      r[`WF(REQ, SUM)] = sum;
      r[`WF(REQ, MXR)] = mxr;
      return r;
    endfunction
    function void program_pte(u64 address, u64 value, bit error = 0);
      mmu_ref_program(model, address, value, error);
    endfunction
    function u64 pte(u64 address);
      longint unsigned value;
      int unsigned error;
      if (!mmu_ref_read(model, address, value, error))
        `uvm_fatal("FIXTURE", "Attempt to modify an unprogrammed PTE")
      return value;
    endfunction
    function translation golden(req_t r, int mode, u64 root);
      longint unsigned address;
      int unsigned page_fault, access_fault, level;
      translation t = new(current_case);
      if (mmu_ref_translate(
              model,
              r[
              `WF(REQ, VADDR)
              ],
              mode,
              root,
              r[
              `WF(REQ, PRIVILEGE)
              ],
              r[
              `WF(REQ, WRITE)
              ],
              r[
              `WF(REQ, EXECUTE)
              ],
              r[
              `WF(REQ, SUM)
              ],
              r[
              `WF(REQ, MXR)
              ],
              address,
              page_fault,
              access_fault,
              level
          ) != 0)
        `uvm_fatal("FIXTURE", $sformatf("%s uninitialized golden PTE %h", current_case, address))
      t.paddr = address;
      t.page_fault = page_fault;
      t.access_fault = access_fault;
      t.level = level;
      return t;
    endfunction
    function u64 pte_address(u64 va, int level);
      return (64'h12 - level) * 4096 + ((va >> (12 + 9 * level)) & 511) * 8;
    endfunction
    function void prepare(u64 va, int leaf_level, u64 flags = 64'hcf, u64 mapped = 64'h80000000);
      mmu_ref_clear(model);
      for (int level = 2; level >= leaf_level; level--) begin
        if (level == leaf_level)
          program_pte(pte_address(va, level), ((mapped >> 12) << 10) | flags);
        else program_pte(pte_address(va, level), ((64'h13 - level) << 10) | 1);
      end
    endfunction
    task service();
      forever begin
        @(req.sample);
        if (!control.reset) begin
          cycle++;
          if (req.sample.valid && req.sample.ready) accepted++;
          if (req.sample.valid && !req.sample.ready) req_stalls++;
          if (access.sample.valid && !access.sample.ready) access_stalls++;
          if (resp.sample.valid && !resp.sample.ready) resp_stalls++;
          if (result_active && result.sample.valid && result.sample.ready) begin
            result_active  = 0;
            pending_result = 0;
          end
          if (access.sample.valid) begin
            u64 address;
            longint unsigned next_address, value;
            int unsigned error;
            if ($isunknown(access.sample.bits)) `uvm_fatal("PTE_X", "Unknown PTE memory request")
            address = access.sample.bits[`WF(ACCESS, ADDR)];
            if (!mmu_ref_peek(model, next_address) || address !== next_address)
              `uvm_fatal("PTE_SEQUENCE", $sformatf(
                         "%s unexpected PTE address %h (remaining=%0d)",
                         current_case,
                         address,
                         mmu_ref_pending(
                             model
                         )
                         ))
            if (access.sample.bits[
                `WF(ACCESS, WRITE)
                ] || access.sample.bits[
                `WF(ACCESS, DATA)
                ] != 0 || access.sample.bits[
                `WF(ACCESS, MASK)
                ] != 0 || access.sample.bits[
                `WF(ACCESS, ATOMIC)
                ] != 0 || access.sample.bits[
                `WF(ACCESS, ATOMICWORD)
                ] != 0)
              `uvm_fatal("PTE_WRITE", "Svade profile permits only ordinary 64-bit PTE reads")
            if (!mmu_ref_read(model, address, value, error))
              `uvm_fatal("UNINITIALIZED_PTE", "DUT read an unprogrammed memory location")
            if (access.sample.ready) begin
              if (pending_result)
                `uvm_fatal("OUTSTANDING", "Walker issued more than one PTE access")
              if (!mmu_ref_consume(model, address))
                `uvm_fatal("PTE_SEQUENCE", "Reference PTE queue changed before handshake")
              reads++;
              result_packet = '0;
              result_packet[`WF(RESULT, DATA)] = value;
              result_packet[`WF(RESULT, ERROR)] = error;
              due = cycle + 2 + reads % 4;
              pending_result = 1;
            end
          end
          if (resp.sample.valid && resp.sample.ready) begin
            translation actual = new(current_case);
            if ($isunknown(resp.sample.bits))
              `uvm_fatal("RESPONSE_X", "Unknown translation response")
            actual.paddr = resp.sample.bits[`WF(RESP, PADDR)];
            actual.page_fault = resp.sample.bits[`WF(RESP, PAGEFAULT)];
            actual.access_fault = resp.sample.bits[`WF(RESP, ACCESSFAULT)];
            actual.level = resp.sample.bits[`WF(RESP, LEVEL)];
            if (actual.page_fault && actual.access_fault)
              `uvm_fatal("FAULT_KIND", "Fault kinds must be mutually exclusive")
            if ((actual.page_fault || actual.access_fault) && actual.paddr != 0)
              `uvm_fatal("FAULT_PA", "Fault response exposed a physical address")
            scoreboard.actual_export.write(actual);
            returned++;
            if (actual.page_fault) page_faults++;
            else if (actual.access_fault) access_faults++;
            else successes[actual.level]++;
          end
        end
        @(negedge control.clock);
        access.ready = !control.reset && !hold_access && cycle % 5 >= 2;
        resp.ready   = !control.reset && !hold_response && cycle % 7 >= 3;
        if (pending_result && !result_active && cycle >= due) result_active = 1;
        result.valid = result_active;
        result.bits  = result_packet;
      end
    endtask
    task send(req_t r, int mode = 8, u64 root = 16, bit change_inputs = 0);
      @(negedge control.clock);
      control.mode = mode;
      control.root_ppn = root;
      req.bits = r;
      req.valid = 1;
      do @(req.sample); while (!req.sample.ready);
      @(negedge control.clock);
      req.valid = 0;
      if (change_inputs) begin
        control.mode = 0;
        control.root_ppn = 0;
        req.bits = ~r;
      end
    endtask
    task run_case(string name, req_t r, int mode = 8, u64 root = 16, bit change_inputs = 0);
      translation prediction;
      current_case = name;
      if (mmu_ref_pending(model) || pending_result)
        `uvm_fatal("PENDING", "Previous translation not drained")
      prediction = golden(r, mode, root);
      scoreboard.expected_export.write(prediction);
      expected++;
      send(r, mode, root, change_inputs);
      scoreboard.wait_checked(expected);
      @(negedge control.clock);
      if (mmu_ref_pending(model) || pending_result)
        `uvm_fatal("PENDING", "Translation skipped expected PTE reads")
    endtask
    task execute();
      u64 va = 64'h123456789, flags, mapped;
      int permissions[5] = '{1, 3, 4, 5, 7};
      control.reset = 1;
      control.mode = 0;
      control.root_ppn = 0;
      req.valid = 0;
      req.bits = '0;
      resp.ready = 0;
      access.ready = 0;
      result.valid = 0;
      result.bits = '0;
      repeat (5) @(negedge control.clock);
      control.reset = 0;
      fork
        service();
      join_none
      // Explicitly force a stalled PTE request, then hold the response while a
      // second request is pending. Both expected paths use initialized table words.
      prepare(va, 0, 64'h43, 64'h81234000);
      current_case = "back_to_back_read_then_execute_fault";
      scoreboard.expected_export.write(golden(request(va), 8, 16));
      expected++;
      scoreboard.expected_export.write(golden(request(va, 2), 8, 16));
      expected++;
      hold_access   = 1;
      hold_response = 1;
      send(request(va));
      fork
        send(request(va, 2));
        begin
          wait (access.sample.valid);
          repeat (8) @(req.sample);
          hold_access = 0;
          wait (resp.sample.valid);
          repeat (8) @(req.sample);
          hold_response = 0;
        end
      join
      scoreboard.wait_checked(expected);
      @(negedge control.clock);
      if (mmu_ref_pending(model) || pending_result)
        `uvm_fatal("PENDING", "Back-to-back translations failed to drain")
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
          program_pte(pte_address(va, level), pte(pte_address(va, level)
                      ) | (64'h1 << reserved_bit));
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
      mmu_ref_clear(model);
      run_case("root_physical_overflow", request(va), 8, HIGH_PPN);
      run_case("positive_noncanonical", request(64'h0000004000000000));
      run_case("upper_noncanonical", request(64'hffffffbfffffffff));
      for (int operation = 0; operation < 3; operation++) begin
        for (int privilege = 0; privilege < 2; privilege++) begin
          run_case("bare_noncanonical_physical", request(64'h8000000000, operation, privilege), 0,
                   0);
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
      mmu_ref_clear(model);
      program_pte(MAX_PA - 7, 64'hcf);
      run_case("last_physical_pte_word", request(64'hffffffffc0000000), 8, MAX_PA >> 12);
      if(expected!=accepted||expected!=returned||!req_stalls||!access_stalls||!resp_stalls||!page_faults||!access_faults||!successes[0]||!successes[1]||!successes[2])
        `uvm_fatal("SCENARIOS", "Walker scenarios or handshake checks were not exercised")
      `uvm_info(
          "CHECKS",
          $sformatf(
              "Sv39/Svade walker checked=%0d PTEreads=%0d successLevels=%0d/%0d/%0d PF=%0d AF=%0d stalls(req/access/resp)=%0d/%0d/%0d; explicit PTE dictionary, no implicit writes",
              expected, reads, successes[0], successes[1], successes[2], page_faults,
              access_faults, req_stalls, access_stalls, resp_stalls), UVM_LOW)
    endtask
    function void final_phase(uvm_phase phase);
      mmu_ref_destroy(model);
    endfunction
  endclass
endpackage
