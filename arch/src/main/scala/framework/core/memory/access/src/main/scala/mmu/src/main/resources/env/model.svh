class model extends uvm_subscriber #(sample_item);
  `uvm_component_utils(model)
  chandle reference;
  sample_item latest;
  bit active = 0, pending_result = 0, hold_access = 0, hold_response = 0, hold_result = 0;
  longint unsigned result_data;
  bit result_error;
  int due = 0, cycle = 0, accepted = 0, returned = 0, reads = 0, cancelled = 0;
  string current_case;
  uvm_analysis_port #(result_item) expected_ap, actual_ap;
  uvm_analysis_port #(bit) reset_ap;
  covergroup behavior with function sample (
      int privilege, bit write, bit execute, bit hit, int level, bit pf, bit af
  );
    option.per_instance = 1;
    privilege_mode: coverpoint privilege {
      bins user_mode = {0}; bins supervisor = {1}; bins machine_mode = {3};
    }
    write_access: coverpoint write;
    execute_access: coverpoint execute;
    cached_translation: coverpoint hit;
    page_size: coverpoint level iff (!pf && !af) {bins sizes[] = {[0 : 2]};}
    page_fault: coverpoint pf;
    access_fault: coverpoint af;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    expected_ap = new("expected_ap", this);
    actual_ap = new("actual_ap", this);
    reset_ap = new("reset_ap", this);
    behavior = new();
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    reference = mmu_ref_create(`MMU_ADDRESS_BITS, `MMU_TLB_ENTRIES);
  endfunction
  function void write(sample_item t);
    result_item expected, actual;
    longint unsigned paddr, next_address, value, probe_paddr, address;
    int unsigned pf, af, level, error, probe_hit;
    bit req_fire, resp_fire, pte_result;
    latest = t;
    if (t.reset) begin
      reset_ap.write(1);
      if (active) cancelled++;
      active = 0;
      pending_result = 0;
      mmu_ref_reset(reference);
      return;
    end
    cycle++;
    probe_hit = mmu_ref_lookup(
        reference,
        t.inputs[`MMU_IN_LOOKUP_REQ_VADDR_OFFSET+:`MMU_IN_LOOKUP_REQ_VADDR_WIDTH],
        t.inputs[`MMU_IN_LOOKUP_CONFIG_MODE_OFFSET+:`MMU_IN_LOOKUP_CONFIG_MODE_WIDTH],
        t.inputs[`MMU_IN_LOOKUP_CONFIG_ROOTPPN_OFFSET+:`MMU_IN_LOOKUP_CONFIG_ROOTPPN_WIDTH],
        t.inputs[`MMU_IN_LOOKUP_REQ_PRIVILEGE_OFFSET+:`MMU_IN_LOOKUP_REQ_PRIVILEGE_WIDTH],
        t.inputs[`MMU_IN_LOOKUP_REQ_WRITE_OFFSET+:`MMU_IN_LOOKUP_REQ_WRITE_WIDTH],
        t.inputs[`MMU_IN_LOOKUP_REQ_EXECUTE_OFFSET+:`MMU_IN_LOOKUP_REQ_EXECUTE_WIDTH],
        t.inputs[`MMU_IN_LOOKUP_REQ_SUM_OFFSET+:`MMU_IN_LOOKUP_REQ_SUM_WIDTH],
        t.inputs[`MMU_IN_LOOKUP_REQ_MXR_OFFSET+:`MMU_IN_LOOKUP_REQ_MXR_WIDTH],
        probe_paddr
    );
    evidence::check(`MO(t.outputs, LOOKUP_HIT) === 1'(probe_hit), $sformatf(
                    "%s lookup hit mismatch va=%h mode=%0d root=%h expected=%0d got=%0d",
                    current_case,
                    t.inputs[`MMU_IN_LOOKUP_REQ_VADDR_OFFSET+:`MMU_IN_LOOKUP_REQ_VADDR_WIDTH],
                    t.inputs[`MMU_IN_LOOKUP_CONFIG_MODE_OFFSET+:`MMU_IN_LOOKUP_CONFIG_MODE_WIDTH],
                    t.inputs[`MMU_IN_LOOKUP_CONFIG_ROOTPPN_OFFSET+:`MMU_IN_LOOKUP_CONFIG_ROOTPPN_WIDTH],
                    probe_hit,
                    `MO(t.outputs, LOOKUP_HIT)
                    ));
    if (probe_hit)
      evidence::check(`MO(t.outputs, LOOKUP_PADDR) === `MMU_ADDRESS_BITS'(probe_paddr),
                      "Walker lookup returned the wrong physical address");
    req_fire = t.inputs[`MMU_IN_REQ_VALID_OFFSET+:`MMU_IN_REQ_VALID_WIDTH] && `MO(t.outputs,
                                                                                  REQ_READY);
    resp_fire =
    `MO(t.outputs, RESP_VALID)
    && t.inputs[`MMU_IN_RESP_READY_OFFSET+:`MMU_IN_RESP_READY_WIDTH];
    pte_result = t.inputs[`MMU_IN_RESULT_VALID_OFFSET+:`MMU_IN_RESULT_VALID_WIDTH] && `MO(
        t.outputs, RESULT_READY);
    if (resp_fire) begin
      evidence::check(active, "Walker returned an unsolicited translation");
      actual = result_item::type_id::create("actual");
      actual.payload = t.outputs[`MMU_RESP_OFFSET+:`MMU_RESP_BITS];
      actual_ap.write(actual);
      active = 0;
      returned++;
    end
    if (req_fire) begin
      evidence::check(!active, "Walker overlapped address translations");
      if (mmu_ref_translate(
              reference,
              t.inputs[`MMU_IN_REQ_BITS_VADDR_OFFSET+:`MMU_IN_REQ_BITS_VADDR_WIDTH],
              t.inputs[`MMU_IN_CONFIG_MODE_OFFSET+:`MMU_IN_CONFIG_MODE_WIDTH],
              t.inputs[`MMU_IN_CONFIG_ROOTPPN_OFFSET+:`MMU_IN_CONFIG_ROOTPPN_WIDTH],
              t.inputs[`MMU_IN_REQ_BITS_PRIVILEGE_OFFSET+:`MMU_IN_REQ_BITS_PRIVILEGE_WIDTH],
              t.inputs[`MMU_IN_REQ_BITS_WRITE_OFFSET+:`MMU_IN_REQ_BITS_WRITE_WIDTH],
              t.inputs[`MMU_IN_REQ_BITS_EXECUTE_OFFSET+:`MMU_IN_REQ_BITS_EXECUTE_WIDTH],
              t.inputs[`MMU_IN_REQ_BITS_SUM_OFFSET+:`MMU_IN_REQ_BITS_SUM_WIDTH],
              t.inputs[`MMU_IN_REQ_BITS_MXR_OFFSET+:`MMU_IN_REQ_BITS_MXR_WIDTH],
              paddr,
              pf,
              af,
              level
          ) != 0)
        `uvm_fatal("PTE_FIXTURE", $sformatf(
                   "%s requires an unprogrammed PTE at %h", current_case, paddr))
      expected = result_item::type_id::create("expected");
      expected.payload = '0;
      expected.payload[`MMU_OUT_RESP_BITS_PADDR_OFFSET-`MMU_RESP_OFFSET+:`MMU_OUT_RESP_BITS_PADDR_WIDTH]=paddr;
      expected.payload[`MMU_OUT_RESP_BITS_PAGEFAULT_OFFSET-`MMU_RESP_OFFSET] = pf;
      expected.payload[`MMU_OUT_RESP_BITS_ACCESSFAULT_OFFSET-`MMU_RESP_OFFSET] = af;
      expected.payload[`MMU_OUT_RESP_BITS_LEVEL_OFFSET-`MMU_RESP_OFFSET+:2] = level;
      expected_ap.write(expected);
      active = 1;
      accepted++;
      behavior.sample(t.inputs[`MMU_IN_REQ_BITS_PRIVILEGE_OFFSET+:`MMU_IN_REQ_BITS_PRIVILEGE_WIDTH],
                      t.inputs[`MMU_IN_REQ_BITS_WRITE_OFFSET+:`MMU_IN_REQ_BITS_WRITE_WIDTH],
                      t.inputs[`MMU_IN_REQ_BITS_EXECUTE_OFFSET+:`MMU_IN_REQ_BITS_EXECUTE_WIDTH],
                      mmu_ref_cached(reference), level, pf, af);
    end
    if (pte_result) begin
      evidence::check(pending_result, "Walker accepted an unsolicited PTE result");
      mmu_ref_result(reference);
      pending_result = 0;
    end
    if (`MO(t.outputs, ACCESS_VALID)) begin
      address = `MO(t.outputs, ACCESS_BITS_ADDR);
      evidence::check(mmu_ref_peek(reference, next_address
                      ) && address === next_address && address[2:0] == 0, $sformatf(
                      "%s unexpected PTE access %h", current_case, address));
      if (t.inputs[`MMU_IN_ACCESS_READY_OFFSET+:`MMU_IN_ACCESS_READY_WIDTH]) begin
        evidence::check(!pending_result, "Walker overlapped PTE accesses");
        if (!mmu_ref_read(reference, address, value, error))
          `uvm_fatal("PTE_FIXTURE", "Walker read an unprogrammed PTE")
        evidence::check(mmu_ref_consume(reference, address), "Walker skipped an expected PTE read");
        result_data  = value;
        result_error = error;
        reads++;
        due = cycle + 2 + reads % 4;
        pending_result = 1;
      end
    end
    if (t.inputs[`MMU_IN_FLUSH_OFFSET+:`MMU_IN_FLUSH_WIDTH]) mmu_ref_flush(reference);
  endfunction
  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    evidence::check(!active && !pending_result && mmu_ref_pending(reference
                    ) == 0 && accepted == returned + cancelled,
                    "Walker transactions did not drain");
  endfunction
  function void final_phase(uvm_phase phase);
    mmu_ref_destroy(reference);
    super.final_phase(phase);
  endfunction
endclass
