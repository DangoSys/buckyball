`include "virtual_cache_config.svh"
`define VF(K, F) `VM_``K``_``F``_OFFSET +: `VM_``K``_``F``_WIDTH
package virtual_cache_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  typedef bit [63:0] u64;
  typedef logic [`VM_REQ_WIDTH-1:0] req_t;
  typedef logic [`VM_MEMRESP_WIDTH-1:0] mem_t;
  import "DPI-C" function chandle mmu_ref_create(input int unsigned address_bits);
  import "DPI-C" function void mmu_ref_destroy(input chandle model);
  import "DPI-C" function void mmu_ref_program(
    input chandle model,
    input longint unsigned address,
    value,
    input int unsigned error
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
  import "DPI-C" function void cpu_mem_ref_prepare(
    input longint unsigned addr,
    input int unsigned size,
    write,
    input longint unsigned data,
    input int unsigned atomic,
    cacheable,
    normal,
    address_bits,
    output int unsigned target,
    misaligned,
    access_fault,
    output longint unsigned bus_addr,
    bus_data,
    output int unsigned bus_mask,
    atomic_word
  );
  import "DPI-C" function longint unsigned cpu_mem_ref_result(
    input longint unsigned addr,
    input int unsigned size,
    write,
    signed_load,
    atomic,
    cacheable,
    input longint unsigned raw,
    input int unsigned error
  );
  class result_item extends uvm_sequence_item;
    bit [5:0] tag;
    u64 data;
    bit misaligned, page_fault, access_fault;
    `uvm_object_utils_begin(result_item)
      `uvm_field_int(tag, UVM_DEFAULT)
      `uvm_field_int(data, UVM_DEFAULT)
      `uvm_field_int(misaligned, UVM_DEFAULT)
      `uvm_field_int(page_fault, UVM_DEFAULT)
      `uvm_field_int(access_fault, UVM_DEFAULT)
    `uvm_object_utils_end
    function new(string name = "result_item");
      super.new(name);
    endfunction
  endclass
  class protocol_test extends ip_test;
    `uvm_component_utils(protocol_test)
    virtual virtual_cache_if control;
    virtual stream_if #(`VM_AUTH_WIDTH) auth_req;
    virtual stream_if #(1) auth_resp;
    virtual stream_if #(`VM_REQ_WIDTH) req;
    virtual stream_if #(`VM_RESP_WIDTH) resp;
    virtual stream_if #(`VM_MEMREQ_WIDTH) mem_req;
    virtual stream_if #(`VM_MEMRESP_WIDTH) mem_resp;
    virtual stream_if #(`VM_UNCACHED_WIDTH) uncached_req;
    virtual stream_if #(`VM_URESP_WIDTH) uncached_resp;
    in_order_scoreboard #(result_item) scoreboard;
    chandle mmu;
    bit [511:0] backing[u64], architectural[u64];
    bit [7:0] mmio_backing[u64], mmio_arch[u64];
    bit line_error[u64], mmio_error[u64];
    int evictions[u64];
    typedef struct {
      mem_t bits;
      int   due;
    } memory_entry;
    memory_entry pending[$];
    bit mem_active = 0, uncached_active = 0, uncached_pending = 0;
    mem_t memory_packet;
    logic [`VM_URESP_WIDTH-1:0] uncached_packet;
    int
        uncached_due,
        cycle = 0,
        expected = 0,
        accepted = 0,
        returned = 0,
        pte_reads = 0,
        data_accesses = 0,
        uncached_accesses = 0;
    int
        memory_reads = 0,
        memory_writes = 0,
        req_stalls = 0,
        resp_stalls = 0,
        mem_stalls = 0,
        uncached_stalls = 0;
    int page_faults = 0, access_faults = 0, misalignments = 0, sc_successes = 0, sc_evictions = 0;
    int invalid_case = -1;
    bit deny_final = 0, deny_pte = 0;
    u64 denied_pte = 0;
    bit
        final_queried = 0,
        auth_pending = 0,
        auth_allow = 0,
        auth_is_pte = 0,
        pte_permit = 0,
        final_permit = 0;
    u64 auth_address = 0, pte_permitted_address = 0;
    int auth_due = 0, auth_queries = 0, auth_denials = 0, auth_stalls = 0;

    bit
        hold_response = 0,
        case_active = 0,
        expect_translation = 0,
        expect_physical = 0,
        expect_cache = 0,
        expect_uncached = 0;
    bit saw_translation = 0, saw_physical = 0, saw_cache = 0, saw_uncached = 0, commit_write = 0;
    int uncached_pte_reads = 0;
    int expected_region = 0, commit_bytes = 0;
    u64 expected_pa = 0, commit_data = 0;
    logic [`VM_TRANSLATION_WIDTH-1:0] translation_expected;
    logic [`VM_PHYSICAL_WIDTH-1:0] physical_expected;
    logic [`VM_CACHE_WIDTH-1:0] cache_expected;
    logic [`VM_UNCACHED_WIDTH-1:0] uncached_expected;
    req_t current;
    string current_case;
    u64 next_table = `VM_DDR_BASE + 64'h10000;
    localparam u64 DATA = `VM_DDR_BASE + 64'h800180;
    localparam u64 EVICT_DATA = `VM_DDR_BASE + 64'h810000;
    localparam u64 ERROR_DATA = `VM_DDR_BASE + 64'h900180;
    localparam u64 MAX_PA = (64'h1 << `VM_CACHE_ADDR_WIDTH) - 1;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      timeout = 2ms;
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      scoreboard = in_order_scoreboard#(result_item)::type_id::create("scoreboard", this);
      mmu = mmu_ref_create(`VM_CACHE_ADDR_WIDTH);
      if (!uvm_config_db#(virtual stream_if #(`VM_AUTH_WIDTH))::get(
              this, "", "auth_req", auth_req
          ) || !uvm_config_db#(virtual stream_if #(1))::get(
              this, "", "auth_resp", auth_resp
          ) || !uvm_config_db#(virtual virtual_cache_if)::get(
              this, "", "control", control
          ) || !uvm_config_db#(virtual stream_if #(`VM_REQ_WIDTH))::get(
              this, "", "req", req
          ) || !uvm_config_db#(virtual stream_if #(`VM_RESP_WIDTH))::get(
              this, "", "resp", resp
          ) || !uvm_config_db#(virtual stream_if #(`VM_MEMREQ_WIDTH))::get(
              this, "", "mem_req", mem_req
          ) || !uvm_config_db#(virtual stream_if #(`VM_MEMRESP_WIDTH))::get(
              this, "", "mem_resp", mem_resp
          ) || !uvm_config_db#(virtual stream_if #(`VM_UNCACHED_WIDTH))::get(
              this, "", "uncached_req", uncached_req
          ) || !uvm_config_db#(virtual stream_if #(`VM_URESP_WIDTH))::get(
              this, "", "uncached_resp", uncached_resp
          ))
        `uvm_fatal("VIF", "Virtual cache interfaces missing")
      void'(uvm_config_db#(int)::get(this, "", "invalid_case", invalid_case));
    endfunction
    function int region(u64 address, int bytes);
      if (address > MAX_PA || bytes - 1 > MAX_PA - address) return 0;
      if(address>=`VM_READ_ONLY_BASE&&address+bytes<=`VM_WRITE_ONLY_BASE+`VM_WRITE_ONLY_BYTES)
        return 1;
      if(address>=`VM_DDR_BASE&&address-`VM_DDR_BASE<`VM_DDR_BYTES&&bytes<=`VM_DDR_BYTES-(address-`VM_DDR_BASE))
        return 1;
      if(address>=`VM_MMIO_BASE&&address-`VM_MMIO_BASE<`VM_MMIO_BYTES&&bytes<=`VM_MMIO_BYTES-(address-`VM_MMIO_BASE))
        return 2;
      if(address>=`VM_SHORT_MMIO_BASE&&address-`VM_SHORT_MMIO_BASE<`VM_SHORT_MMIO_BYTES&&bytes<=`VM_SHORT_MMIO_BYTES-(address-`VM_SHORT_MMIO_BASE))
        return 2;
      if(address>=`VM_NORMAL_RAM_BASE&&address+bytes<=`VM_NORMAL_RAM_BASE+`VM_NORMAL_RAM_BYTES)
        return 3;
      return 0;
    endfunction
    function void initialize_line(u64 address, bit [511:0] value);
      u64 base = address & ~64'h3f;
      if (backing.exists(base)) `uvm_fatal("FIXTURE", "Attempt to reinitialize a live DDR line")
      backing[base] = value;
      architectural[base] = value;
    endfunction
    function void program_pte(u64 address, u64 value, bit error = 0);
      u64 base = address & ~64'h3f;
      if (region(address, 8) == 1 || region(address, 8) == 3) begin
        if (!backing.exists(base)) initialize_line(base, '0);
        backing[base][int'(address&63)*8+:64] = value;
        architectural[base] = backing[base];
        if (error) line_error[base] = 1;
      end else if (!error) `uvm_fatal("FIXTURE", "Non-DDR PTE must be an explicit access error")
      mmu_ref_program(mmu, address, value, error);
    endfunction
    function u64 mapping(u64 va, u64 pa, u64 flags = 64'hcf, int error_level = -1);
      u64 root = next_table;
      next_table += 64'h3000;
      for (int level = 2; level >= 0; level--) begin
        u64 table = root + (2 - level) * 4096;
        u64 address = table + ((va >> (12 + 9 * level)) & 511) * 8;
        program_pte(address,
                    level == 0 ? ((pa >> 12) << 10) | flags : (((table + 4096) >> 12) << 10) | 1,
                    level == error_level);
      end
      return root >> 12;
    endfunction
    function req_t command(u64 address, int size = 3, bit write = 0, bit signed_load = 0,
                           u64 data = 0, int atomic = 0, int mode = 0, u64 root = 0,
                           int privilege = 1, bit sum = 0, bit mxr = 0, bit execute = 0);
      req_t r = '0;
      r[`VF(REQ, EXECUTE)] = execute;
      r[`VF(REQ, VADDR)] = address;
      r[`VF(REQ, SIZE)] = size;
      r[`VF(REQ, WRITE)] = write;
      r[`VF(REQ, SIGNED)] = signed_load;
      r[`VF(REQ, DATA)] = data;
      r[`VF(REQ, ATOMIC)] = atomic;
      r[`VF(REQ, SATPMODE)] = mode;
      r[`VF(REQ, ROOTPPN)] = root;
      r[`VF(REQ, PRIVILEGE)] = privilege;
      r[`VF(REQ, SUM)] = sum;
      r[`VF(REQ, MXR)] = mxr;
      return r;
    endfunction
    function u64 read_arch(u64 address, int bytes, int kind);
      u64 value = 0;
      for (int i = 0; i < bytes; i++) begin
        u64 a = address + i;
        if (kind == 1 || kind == 3) begin
          if (!architectural.exists(a & ~64'h3f))
            `uvm_fatal("FIXTURE", $sformatf("Uninitialized architecture DDR %h", a))
          value |= u64'(architectural[a&~64'h3f][int'(a&63)*8+:8]) << (i * 8);
        end else begin
          if (!mmio_arch.exists(a))
            `uvm_fatal("FIXTURE", $sformatf("Uninitialized architecture MMIO %h", a))
          value |= u64'(mmio_arch[a]) << (i * 8);
        end
      end
      return value;
    endfunction
    function void predict(req_t r, int sc_result);
      int unsigned target, misaligned, af, mask, word, pf, level;
      longint unsigned pa, bus_addr, bus_data;
      u64 raw, old, operand, next_value, saved_pte;
      int size, atom, bytes;
      bit write, failed;
      result_item answer = new(current_case);
      if (mmu_ref_pending(mmu))
        `uvm_fatal("PTE_PENDING", "Previous reference PTE chain not drained")
      current = r;
      case_active = 1;
      commit_write = 0;
      pte_permit = 0;
      final_permit = 0;
      final_queried = 0;
      expect_translation = 0;
      expect_physical = 0;
      expect_cache = 0;
      expect_uncached = 0;
      saw_translation = 0;
      saw_physical = 0;
      saw_cache = 0;
      saw_uncached = 0;
      translation_expected = '0;
      physical_expected = '0;
      cache_expected = '0;
      uncached_expected = '0;
      size = r[`VF(REQ, SIZE)];
      bytes = 1 << size;
      atom = r[`VF(REQ, ATOMIC)];
      write = r[`VF(REQ, WRITE)];
      answer.tag = r[`VF(REQ, TAG)];
      // Before translation only alignment is meaningful; use physical reference
      // again with the translated PA for its actual range/classification result.
      cpu_mem_ref_prepare(r[`VF(REQ, VADDR)], size, write, r[`VF(REQ, DATA)], atom, 1, 1,
                          `VM_CACHE_ADDR_WIDTH, target, misaligned, af, bus_addr, bus_data, mask,
                          word);
      answer.misaligned = misaligned;
      if (!misaligned) begin
        expect_translation = 1;
        if (deny_pte) begin
          saved_pte = read_arch(denied_pte, 8, 1);
          mmu_ref_program(mmu, denied_pte, saved_pte, 1);
        end
        if (mmu_ref_translate(
                mmu,
                r[
                `VF(REQ, VADDR)
                ],
                r[
                `VF(REQ, SATPMODE)
                ],
                r[
                `VF(REQ, ROOTPPN)
                ],
                r[
                `VF(REQ, PRIVILEGE)
                ],
                write || (atom >= 1 && atom <= 9) || atom == 11,
                r[
                `VF(REQ, EXECUTE)
                ],
                r[
                `VF(REQ, SUM)
                ],
                r[
                `VF(REQ, MXR)
                ],
                pa,
                pf,
                af,
                level
            ) != 0)
          `uvm_fatal("FIXTURE", $sformatf("Reference requested an unprogrammed PTE %h", pa))
        if (deny_pte) mmu_ref_program(mmu, denied_pte, saved_pte, 0);
        translation_expected[`VF(TRANSLATION, PADDR)] = pa;
        translation_expected[`VF(TRANSLATION, PAGEFAULT)] = pf;
        translation_expected[`VF(TRANSLATION, ACCESSFAULT)] = af;
        translation_expected[`VF(TRANSLATION, LEVEL)] = level;
        answer.page_fault = pf;
        answer.access_fault = af;
        if (!pf && !af) begin
          expected_pa = pa;
          expected_region = region(pa, bytes);
          if(deny_final||(pa>=`VM_READ_ONLY_BASE&&pa<`VM_READ_ONLY_BASE+64&&(write||atom!=0||r[
              `VF(REQ, EXECUTE)
              ])) || (pa >= `VM_WRITE_ONLY_BASE && pa < `VM_WRITE_ONLY_BASE + 64 &&
                      (!write || atom != 0 || r[
              `VF(REQ, EXECUTE)
              ])) || expected_region == 0 ||
                  (atom != 0 && expected_region != 1 && expected_region != 3) || (r[
              `VF(REQ, EXECUTE)
              ] && expected_region != 1 && expected_region != 3))
            answer.access_fault = 1;
          else begin
            expect_physical = 1;
            physical_expected[`VF(PHYSICAL, ADDR)] = pa;
            physical_expected[`VF(PHYSICAL, TAG)] = r[`VF(REQ, TAG)];
            physical_expected[`VF(PHYSICAL, SIZE)] = size;
            physical_expected[`VF(PHYSICAL, WRITE)] = write;
            physical_expected[`VF(PHYSICAL, SIGNED)] = r[`VF(REQ, SIGNED)];
            physical_expected[`VF(PHYSICAL, DATA)] = r[`VF(REQ, DATA)];
            physical_expected[`VF(PHYSICAL, ATOMIC)] = atom;
            physical_expected[`VF(PHYSICAL, CACHEABLE)] = expected_region == 1;
            physical_expected[`VF(PHYSICAL, NORMAL)] = expected_region == 1 || expected_region == 3;
            cpu_mem_ref_prepare(pa, size, write, r[`VF(REQ, DATA)], atom, expected_region == 1,
                                expected_region == 1 || expected_region == 3, `VM_CACHE_ADDR_WIDTH,
                                target, misaligned, af, bus_addr, bus_data, mask, word);
            answer.access_fault = af;
            if (target != 0) begin
              expect_cache = target == 1;
              expect_uncached = target == 2;
              cache_expected[`VF(CACHE, ADDR)] = bus_addr;
              cache_expected[`VF(CACHE, WRITE)] = write;
              cache_expected[`VF(CACHE, DATA)] = bus_data;
              cache_expected[`VF(CACHE, MASK)] = mask;
              cache_expected[`VF(CACHE, ATOMIC)] = atom;
              cache_expected[`VF(CACHE, ATOMICWORD)] = word;
              uncached_expected[`VF(UNCACHED, ADDR)] = pa;
              uncached_expected[`VF(UNCACHED, TAG)] = r[`VF(REQ, TAG)];
              uncached_expected[`VF(UNCACHED, SIZE)] = size;
              uncached_expected[`VF(UNCACHED, WRITE)] = write;
              uncached_expected[`VF(UNCACHED, DATA)] = r[`VF(REQ, DATA)];
              uncached_expected[`VF(UNCACHED, ATOMIC)] = atom;
              uncached_expected[
              `VF(UNCACHED, NORMAL)
              ] = expected_region == 1 || expected_region == 3;
              failed = expected_region != 2 ?
                  (line_error.exists(pa & ~64'h3f) && line_error[pa&~64'h3f]) :
                  (mmio_error.exists(pa) && mmio_error[pa]);
              answer.access_fault = failed;
              if (!failed) begin
                old = read_arch(pa, bytes, expected_region);
                operand = r[`VF(REQ, DATA)];
                next_value = operand;
                if (atom != 0) begin
                  raw = (bytes == 4 && old[31]) ? old | 64'hffffffff00000000 : old;
                  case (atom)
                    1: next_value = operand;
                    2: next_value = old + operand;
                    3: next_value = old ^ operand;
                    4: next_value = old & operand;
                    5: next_value = old | operand;
                    6:
                    next_value = bytes == 4 ?
                        ($signed(old[31:0]) < $signed(operand[31:0]) ? old : operand) :
                        ($signed(old) < $signed(operand) ? old : operand);
                    7:
                    next_value = bytes == 4 ?
                        ($signed(old[31:0]) > $signed(operand[31:0]) ? old : operand) :
                        ($signed(old) > $signed(operand) ? old : operand);
                    8:
                    next_value=bytes==4?(old[31:0]<operand[31:0]?old:operand):(old<operand?old:operand);
                    9:
                    next_value=bytes==4?(old[31:0]>operand[31:0]?old:operand):(old>operand?old:operand);
                    10: next_value = old;
                    11: begin
                      if (sc_result < 0 || sc_result > 1)
                        `uvm_fatal("SC_ORACLE",
                                   "SC requires an explicit architectural scenario outcome")
                      raw = sc_result;
                    end
                    default: `uvm_fatal("ATOMIC", "Unsupported test operation")
                  endcase
                end else raw = target == 1 ? read_arch(pa & ~64'h7, 8, 1) : old;
                answer.data = cpu_mem_ref_result(pa, size, write, r[`VF(REQ, SIGNED)], atom,
                                                 target == 1, raw, 0);
                commit_write = write || (atom >= 1 && atom <= 9) || (atom == 11 && sc_result == 0);
                commit_bytes = bytes;
                commit_data = next_value;
              end
            end
          end
        end
      end
      scoreboard.expected_export.write(answer);
      expected++;
    endfunction
    function void observations();
      if (control.sample.eviction_valid) evictions[u64'(control.sample.eviction_addr)]++;
      if (control.sample.cache_valid) begin
        u64 address;
        longint unsigned next_pte;
        logic [`VM_CACHE_WIDTH-1:0] wanted;
        if (!case_active || $isunknown({control.sample.cache_pte, control.sample.cache_bits}))
          `uvm_fatal("CACHE_EVENT", "Unknown or unsolicited cache operation")
        address = control.sample.cache_bits[`VF(CACHE, ADDR)];
        if (control.sample.cache_pte) begin
          if (!pte_permit || pte_permitted_address != address || region(
                  address, 8
              ) != 1 || !mmu_ref_peek(
                  mmu, next_pte
              ) || address != next_pte)
            `uvm_fatal("PTE_CHAIN", $sformatf(
                       "%s wrong real-cache PTE address %h", current_case, address))
          wanted = '0;
          wanted[`VF(CACHE, ADDR)] = next_pte;
          if (control.sample.cache_bits !== wanted)
            `uvm_fatal("PTE_WRITE",
                       "PTE access must be an ordinary cache read, never A/D writeback")
          if (!mmu_ref_consume(mmu, address))
            `uvm_fatal("PTE_CHAIN", "PTE reference queue mismatch")
          pte_permit = 0;
          pte_reads++;
        end else begin
          if(!expect_cache||saw_cache||!saw_translation||!saw_physical||control.sample.cache_bits!==cache_expected)
            `uvm_fatal("CACHE_DATA", $sformatf(
                       "%s wrong PA/lane/mask/operand/atomic cache request", current_case))
          saw_cache = 1;
          data_accesses++;
        end
      end
      if (control.sample.translation_valid) begin
        longint unsigned denied;
        if (!expect_translation || saw_translation || $isunknown(
                control.sample.translation_bits
            ) || control.sample.translation_bits !== translation_expected)
          `uvm_fatal("TRANSLATION", $sformatf(
                     "%s wrong translation result got=%h expected=%h",
                     current_case,
                     control.sample.translation_bits,
                     translation_expected
                     ))
        if (mmu_ref_pending(mmu)) begin
          if (!mmu_ref_peek(
                  mmu, denied
              ) || (region(
                  denied, 8
              ) != 2 && region(
                  denied, 8
              ) != 0 && (!deny_pte || denied != denied_pte)) || !translation_expected[
              `VF(TRANSLATION, ACCESSFAULT)
              ])
            `uvm_fatal("PTE_CHAIN", "Translation skipped a required cache PTE read")
          if (!mmu_ref_consume(mmu, denied) || mmu_ref_pending(mmu))
            `uvm_fatal("PMA", "Unexpected denied-PTE reference chain")
        end
        saw_translation = 1;
      end
      if (control.sample.physical_valid) begin
        if (!final_permit || !expect_physical || saw_physical || !saw_translation || $isunknown(
                control.sample.physical_bits
            ) || control.sample.physical_bits !== physical_expected)
          `uvm_fatal("PHYSICAL", $sformatf(
                     "%s PA classification or captured command changed", current_case))
        saw_physical = 1;
      end
    endfunction
    task service();
      forever begin
        @(control.sample);
        if (!control.sample.reset) begin
          cycle++;
          if (req.sample.valid && req.sample.ready) accepted++;
          if (req.sample.valid && !req.sample.ready) req_stalls++;
          if (resp.sample.valid && !resp.sample.ready) resp_stalls++;
          if (mem_req.sample.valid && !mem_req.sample.ready) mem_stalls++;
          if (uncached_req.sample.valid && !uncached_req.sample.ready) uncached_stalls++;
          if (auth_req.sample.valid && !auth_req.sample.ready) auth_stalls++;
          if (auth_resp.sample.valid && auth_resp.sample.ready) begin
            if (!auth_pending) `uvm_fatal("AUTH_RESPONSE", "Authorization response without query")
            if (auth_allow) begin
              if (auth_is_pte) begin
                pte_permit = 1;
                pte_permitted_address = auth_address;
              end else final_permit = 1;
            end else auth_denials++;
            auth_pending = 0;
          end
          if (auth_req.sample.valid && auth_req.sample.ready) begin
            logic [`VM_AUTH_WIDTH-1:0] wanted = '0;
            longint unsigned pte;
            if (auth_pending || $isunknown(auth_req.sample.bits))
              `uvm_fatal("AUTH_QUERY", "Overlapping or unknown authorization query")
            auth_is_pte = auth_req.sample.bits[`VF(AUTH, ISPTE)];
            if (auth_is_pte) begin
              if (!mmu_ref_peek(mmu, pte)) `uvm_fatal("AUTH_PTE", "Unsolicited PTE authorization")
              wanted[`VF(AUTH, PADDR)] = pte;
              wanted[`VF(AUTH, SIZE)] = 3;
              wanted[`VF(AUTH, READ)] = 1;
              wanted[`VF(AUTH, PRIVILEGE)] = 1;
              wanted[`VF(AUTH, ISPTE)] = 1;
            end else begin
              if (final_queried) `uvm_fatal("AUTH_FINAL", "Repeated final query")
              final_queried = 1;
              if (!saw_translation || translation_expected[
                  `VF(TRANSLATION, PAGEFAULT)
                  ] || translation_expected[
                  `VF(TRANSLATION, ACCESSFAULT)
                  ])
                `uvm_fatal("AUTH_FINAL", "Final authorization before valid translation")
              wanted[`VF(AUTH, PADDR)] = translation_expected[`VF(TRANSLATION, PADDR)];
              wanted[`VF(AUTH, SIZE)] = current[`VF(REQ, SIZE)];
              wanted[
              `VF(AUTH, READ)
              ] = !current[
              `VF(REQ, EXECUTE)
              ] && !current[
              `VF(REQ, WRITE)
              ] && current[
              `VF(REQ, ATOMIC)
              ] != 11;
              wanted[
              `VF(AUTH, WRITE)
              ] = current[
              `VF(REQ, WRITE)
              ] || (current[
              `VF(REQ, ATOMIC)
              ] != 0 && current[
              `VF(REQ, ATOMIC)
              ] != 10);
              wanted[`VF(AUTH, EXECUTE)] = current[`VF(REQ, EXECUTE)];
              wanted[`VF(AUTH, PRIVILEGE)] = current[`VF(REQ, PRIVILEGE)];
            end
            if (auth_req.sample.bits !== wanted)
              `uvm_fatal("AUTH_FIELDS", $sformatf(
                         "%s got=%h expected=%h", current_case, auth_req.sample.bits, wanted))
            auth_address = auth_req.sample.bits[`VF(AUTH, PADDR)];
            auth_allow = !(auth_is_pte ? (deny_pte && auth_address == denied_pte) : deny_final);
            auth_pending = 1;
            auth_due = cycle + 3 + (auth_queries % 5);
            auth_queries++;
          end
          observations();
          if (mem_active && mem_resp.sample.valid && mem_resp.sample.ready) mem_active = 0;
          if (uncached_active && uncached_resp.sample.valid && uncached_resp.sample.ready) begin
            uncached_active  = 0;
            uncached_pending = 0;
          end
          if (mem_req.sample.valid && mem_req.sample.ready) begin
            memory_entry e;
            u64 address;
            bit write, failed;
            bit [511:0] line;
            if ($isunknown(
                    {
                      mem_req.sample.bits[`VF(MEMREQ, ID)],
                      mem_req.sample.bits[`VF(MEMREQ, ADDR)],
                      mem_req.sample.bits[`VF(MEMREQ, WRITE)]
                    }
                ))
              `uvm_fatal("DDR_X", "Unknown DDR request")
            address = mem_req.sample.bits[`VF(MEMREQ, ADDR)];
            write   = mem_req.sample.bits[`VF(MEMREQ, WRITE)];
            if (address[5:0] != 0 || region(address, 64) != 1 || !backing.exists(address))
              `uvm_fatal("DDR_DICTIONARY", $sformatf("Unexpected/uninitialized DDR line %h", address
                         ))
            failed = !write && line_error.exists(address) && line_error[address];
            if (write) begin
              if ($isunknown(
                      {
                        mem_req.sample.bits[`VF(MEMREQ, DATA)],
                        mem_req.sample.bits[`VF(MEMREQ, MASK)]
                      }
                  ))
                `uvm_fatal("DDR_WRITE_X", "Unknown DDR write")
              line = mem_req.sample.bits[`VF(MEMREQ, DATA)];
              if (mem_req.sample.bits[`VF(MEMREQ, MASK)] !== '1 || line !== architectural[address])
                `uvm_fatal("DDR_WRITE", "Cache writeback corrupted full initialized line")
              backing[address] = line;
              memory_writes++;
            end else memory_reads++;
            e.bits = '0;
            e.bits[`VF(MEMRESP, ID)] = mem_req.sample.bits[`VF(MEMREQ, ID)];
            e.bits[`VF(MEMRESP, DATA)] = backing[address];
            e.bits[`VF(MEMRESP, ERROR)] = failed;
            e.due = cycle + 5 + (int'(e.bits[`VF(MEMRESP, ID)]) % 4);
            pending.push_back(e);
          end
          if (uncached_req.sample.valid && uncached_req.sample.ready) begin
            u64 address, value, next_pte;
            int bytes, kind;
            bit failed, write, is_pte;
            address = uncached_req.sample.bits[`VF(UNCACHED, ADDR)];
            bytes = 1 << uncached_req.sample.bits[`VF(UNCACHED, SIZE)];
            write = uncached_req.sample.bits[`VF(UNCACHED, WRITE)];
            kind = region(address, bytes);
            is_pte = !saw_physical && mmu_ref_pending(mmu) != 0;
            if (uncached_pending || $isunknown(uncached_req.sample.bits))
              `uvm_fatal("UNCACHED_OWNER", "Overlapping or unknown uncached request")
            if (is_pte) begin
              logic [`VM_UNCACHED_WIDTH-1:0] wanted = '0;
              if (!pte_permit || pte_permitted_address != address || kind != 3 || !mmu_ref_peek(
                      mmu, next_pte
                  ) || next_pte != address)
                `uvm_fatal("PTE_CHAIN", "Wrong authorized uncached PTE address")
              wanted[`VF(UNCACHED, ADDR)] = address;
              wanted[`VF(UNCACHED, TAG)] = current[`VF(REQ, TAG)];
              wanted[`VF(UNCACHED, SIZE)] = 3;
              wanted[`VF(UNCACHED, NORMAL)] = 1;
              if (uncached_req.sample.bits !== wanted || !mmu_ref_consume(mmu, address))
                `uvm_fatal("PTE_REQUEST", "Uncached PTE contract mismatch")
              pte_permit = 0;
              pte_reads++;
              uncached_pte_reads++;
            end else begin
              if(!expect_uncached||saw_uncached||!saw_physical||uncached_req.sample.bits!==uncached_expected)
                `uvm_fatal("UNCACHED_REQUEST", $sformatf(
                           "%s unexpected uncached request", current_case))
              saw_uncached = 1;
            end
            if (kind != 2 && kind != 3) `uvm_fatal("UNCACHED_REGION", "Invalid uncached region")
            failed = kind == 3 ? (line_error.exists(address & ~64'h3f) && line_error[address&~64'h3f
                                  ]) : (mmio_error.exists(address) && mmio_error[address]);
            value = 0;
            for (int i = 0; i < bytes; i++) begin
              if (kind == 3) begin
                if (!backing.exists(address & ~64'h3f))
                  `uvm_fatal("RAM_DICTIONARY", "Uninitialized normal RAM")
                value |= u64'(backing[address&~64'h3f][int'((address+i)&63)*8+:8]) << (i * 8);
                if (write && !failed)
                  backing[address&~64'h3f][int'((address+i)&63)*8+:8]=uncached_req.sample.bits[`VM_UNCACHED_DATA_OFFSET+i*8+:8];
              end else begin
                if (!mmio_backing.exists(address + i))
                  `uvm_fatal("MMIO_DICTIONARY", "Uninitialized device")
                value |= u64'(mmio_backing[address+i]) << (i * 8);
                if (write && !failed)
                  mmio_backing[address+i]=uncached_req.sample.bits[`VM_UNCACHED_DATA_OFFSET+i*8+:8];
              end
            end
            uncached_packet = '0;
            uncached_packet[`VF(URESP, TAG)] = uncached_req.sample.bits[`VF(UNCACHED, TAG)];
            uncached_packet[`VF(URESP, DATA)] = write ? 0 : value;
            uncached_packet[`VF(URESP, ERROR)] = failed;
            uncached_due = cycle + 7;
            uncached_pending = 1;
            uncached_accesses++;
          end
          if (resp.sample.valid && resp.sample.ready) begin
            result_item actual = new(current_case);
            if (!case_active || $isunknown(resp.sample.bits))
              `uvm_fatal("RESULT", "Unknown or unsolicited virtual result")
            actual.tag = resp.sample.bits[`VF(RESP, TAG)];
            actual.data = resp.sample.bits[`VF(RESP, DATA)];
            actual.misaligned = resp.sample.bits[`VF(RESP, MISALIGNED)];
            actual.page_fault = resp.sample.bits[`VF(RESP, PAGEFAULT)];
            actual.access_fault = resp.sample.bits[`VF(RESP, ACCESSFAULT)];
            if(saw_translation!=expect_translation||saw_physical!=expect_physical||saw_cache!=expect_cache||saw_uncached!=expect_uncached||mmu_ref_pending(
                    mmu
                ))
              `uvm_fatal("SIDE_EFFECT",
                         "Fault suppressed or successful operation skipped a required stage")
            if (auth_pending || final_queried != (expect_translation && !translation_expected[
                `VF(TRANSLATION, PAGEFAULT)
                ] && !translation_expected[
                `VF(TRANSLATION, ACCESSFAULT)
                ]))
              `uvm_fatal("AUTH_DRAIN", "Missing final authorization or response before decision")
            scoreboard.actual_export.write(actual);
            returned++;
            case_active = 0;
            if (actual.page_fault) page_faults++;
            if (actual.access_fault) access_faults++;
            if (actual.misaligned) misalignments++;
            if (commit_write)
              for (int i = 0; i < commit_bytes; i++) begin
                u64 a = expected_pa + i;
                if (expected_region == 1 || expected_region == 3)
                  architectural[a&~64'h3f][int'(a&63)*8+:8] = commit_data[i*8+:8];
                else mmio_arch[a] = commit_data[i*8+:8];
              end
          end
        end
        @(negedge control.clock);
        auth_req.ready = !control.reset && !auth_pending && cycle % 7 >= 4;
        auth_resp.valid = !control.reset && auth_pending && cycle >= auth_due;
        auth_resp.bits = auth_allow;
        mem_req.ready = !control.reset && cycle % 7 >= 2;
        uncached_req.ready = !control.reset && cycle % 9 >= 4;
        resp.ready = !control.reset && !hold_response && cycle % 11 >= 4;
        if (!mem_active)
          for (int i = pending.size() - 1; i >= 0; i--)
          if (pending[i].due <= cycle) begin
            memory_packet = pending[i].bits;
            pending.delete(i);
            mem_active = 1;
            break;
          end
        mem_resp.valid = mem_active;
        mem_resp.bits  = memory_packet;
        if (uncached_pending && cycle >= uncached_due) uncached_active = 1;
        uncached_resp.valid = uncached_active;
        uncached_resp.bits  = uncached_packet;
      end
    endtask
    task run_case(string name, req_t packet, int sc_result = -1, bit perturb = 0, bit drain = 1);
      current_case = name;
      packet[`VF(REQ, TAG)] = expected & 63;
      predict(packet, sc_result);
      @(negedge control.clock);
      req.bits  = packet;
      req.valid = 1;
      do @(req.sample); while (!req.sample.ready);
      @(negedge control.clock);
      req.valid = 0;
      if (perturb) req.bits = ~packet;
      scoreboard.wait_checked(expected);
      if (drain) begin
        do
        @(control.sample);
        while (control.sample.outstanding != 0 || pending.size() || mem_active || uncached_pending);
        repeat (8) @(control.sample);
      end
    endtask
    task execute();
      bit [511:0] pattern;
      u64 root, va, prior;
      int sizes[4] = '{0, 1, 2, 3};
      auth_req.ready = 0;
      auth_resp.valid = 0;
      auth_resp.bits = 0;
      control.conflict_trace = 0;
      control.reset = 1;
      control.active = 0;
      req.valid = 0;
      req.bits = '0;
      resp.ready = 0;
      mem_req.ready = 0;
      mem_resp.valid = 0;
      mem_resp.bits = '0;
      uncached_req.ready = 0;
      uncached_resp.valid = 0;
      uncached_resp.bits = '0;
      repeat (5) @(negedge control.clock);
      control.reset  = 0;
      control.active = 1;
      if (invalid_case >= 0) begin
        req_t bad = command(1);
        bad[`VF(REQ, TAG)] = 23;
        case (invalid_case)
          0: bad[`VF(REQ, SIZE)] = 4;
          1: bad[`VF(REQ, ATOMIC)] = 12;
          2: begin
            bad[`VF(REQ, ATOMIC)] = 2;
            bad[`VF(REQ, WRITE)]  = 1;
          end
          3: begin
            bad[`VF(REQ, ATOMIC)] = 2;
            bad[`VF(REQ, SIZE)]   = 1;
          end
          4: bad[`VF(REQ, SATPMODE)] = 9;
          5: bad[`VF(REQ, PRIVILEGE)] = 2;
          6: begin
            bad[`VF(REQ, EXECUTE)] = 1;
            bad[`VF(REQ, WRITE)]   = 1;
          end
          7: begin
            bad[`VF(REQ, EXECUTE)] = 1;
            bad[`VF(REQ, ATOMIC)]  = 10;
          end
          8: begin
            bad[`VF(REQ, EXECUTE)] = 1;
            bad[`VF(REQ, SIZE)] = 2;
          end
          9: begin
            bad[`VF(REQ, EXECUTE)] = 1;
            bad[`VF(REQ, SIGNED)]  = 1;
          end
          default: `uvm_fatal("CASE", "Unknown virtual negative case")
        endcase
        req.bits  = bad;
        req.valid = 1;
        `uvm_info("INVALID_INPUT", $sformatf(
                                       "Driving illegal virtual command %0d with misaligned VA",
                                       invalid_case), UVM_LOW)
        repeat (10) @(negedge control.clock);
        `uvm_fatal("MISSING_ASSERTION", "Illegal command hidden by early fault response")
      end
      for (int w = 0; w < 8; w++)
        pattern[w*64+:64] = 64'h88776655fedcba80 ^ (64'h0101010101010101 * w);
      initialize_line(`VM_DDR_BASE + 64'h4c0, pattern);
      begin
        bit [511:0] conflict_data = pattern;
        conflict_data[63:0] = 64'hcafebabebeefaa11;
        initialize_line(`VM_DDR_BASE + 64'h40c0, conflict_data);
      end
      initialize_line(`VM_READ_ONLY_BASE, pattern);
      initialize_line(`VM_WRITE_ONLY_BASE, pattern);
      initialize_line(DATA, pattern);
      initialize_line(DATA + 4096, pattern);
      initialize_line(DATA + 8192, pattern);
      initialize_line(EVICT_DATA, pattern);
      initialize_line(ERROR_DATA, pattern);
      initialize_line(`VM_DDR_BASE + `VM_DDR_BYTES - 64, pattern);
      line_error[ERROR_DATA&~64'h3f] = 1;
      for (int i = 0; i < 32; i++) begin
        mmio_arch[`VM_MMIO_BASE+i] = 8'h80 + i;
        mmio_backing[`VM_MMIO_BASE+i] = 8'h80 + i;
      end
      for (int i = 0; i < 5; i++) begin
        mmio_arch[`VM_SHORT_MMIO_BASE+i] = 8'hf0 + i;
        mmio_backing[`VM_SHORT_MMIO_BASE+i] = 8'hf0 + i;
      end
      mmio_error[`VM_MMIO_BASE+24] = 1;
      fork
        service();
      join_none
      repeat (12) @(control.sample);
      // Match the actual Core conflict sequence without idle cycles or HN drain
      // between returned requests: code and data have the same bank/index.
      control.conflict_trace = 1;
      run_case("conflict_code_first", command(
               `VM_DDR_BASE + 64'h4c0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1), -1, 0, 0);
      run_case("conflict_store_W", command(`VM_DDR_BASE + 64'h40c0, 2, 1, 0, 64'h7fffffff), -1, 0,
               0);
      run_case("conflict_code_after_store", command(
               `VM_DDR_BASE + 64'h4d8, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1), -1, 0, 0);
      run_case("conflict_code_AMO_word", command(
               `VM_DDR_BASE + 64'h4e0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1), -1, 0, 0);
      run_case("conflict_AMO_W", command(`VM_DDR_BASE + 64'h40c0, 2, 0, 0, 1, 2), -1, 0, 0);
      run_case("conflict_code_after_AMO", command(
               `VM_DDR_BASE + 64'h4e0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1), -1, 0, 0);
      run_case("conflict_data_readback", command(`VM_DDR_BASE + 64'h40c0), -1, 0, 0);
      do
        @(control.sample);
      while (control.sample.outstanding || pending.size() || mem_active || uncached_pending);
      if(control.conflict_wb!=2||control.conflict_dbid!=2||control.conflict_beats!=4||control.conflict_ack!=6)
        `uvm_fatal("CONFLICT_CHI", $sformatf(
                   "Missing WB/DBID/data/ACK: %0d/%0d/%0d/%0d",
                   control.conflict_wb,
                   control.conflict_dbid,
                   control.conflict_beats,
                   control.conflict_ack
                   ))
      control.conflict_trace = 0;
      `uvm_info(
          "CONFLICT",
          "Code/store/code/AMO.W/code collision completed without idle-cycle insertion; all queues drained",
          UVM_LOW)
      // Hold the second request unchanged until the first response retires, then
      // accept and check it normally; never withdraw a stalled VALID.
      current_case = "single_outstanding_snapshot";
      begin
        req_t first = command(DATA, 0, 0, 1);
        req_t second = command(`VM_MMIO_BASE, 3, 1, 0, 64'hdeadbeef);
        first[`VF(REQ, TAG)]  = 31;
        second[`VF(REQ, TAG)] = 17;
        predict(first, -1);
        hold_response = 1;
        @(negedge control.clock);
        req.bits  = first;
        req.valid = 1;
        do @(req.sample); while (!req.sample.ready);
        @(negedge control.clock);
        req.bits = second;
        wait (resp.sample.valid);
        repeat (12) begin
          @(req.sample);
          if (req.sample.ready)
            `uvm_fatal("OUTSTANDING", "Accepted a second command before retirement");
        end
        @(negedge control.clock);
        hold_response = 0;
        scoreboard.wait_checked(expected);
        @(negedge control.clock);
        current_case = "pending_request_after_retirement";
        predict(second, -1);
        do @(req.sample); while (!req.sample.ready);
        @(negedge control.clock);
        req.valid = 0;
        scoreboard.wait_checked(expected);
        do
        @(control.sample);
        while (control.sample.outstanding || pending.size() || mem_active || uncached_pending);
        repeat (8) @(control.sample);
      end
      foreach (sizes[s])
        for (int offset = 0; offset < 8; offset += (1 << sizes[s]))
          for (int sign_load = 0; sign_load < 2; sign_load++)
            run_case("bare_lane_sign", command(DATA + offset, sizes[s], 0, sign_load));
      foreach (sizes[s]) begin
        run_case("store_mask", command(
                 DATA + (sizes[s] == 0 ? 7 : sizes[s] == 1 ? 2 : sizes[s] == 2 ? 4 : 0),
                 sizes[s],
                 1,
                 0,
                 64'hfe12345689abcdef
                 ));
        run_case("masked_store_readback", command(DATA));
      end
      // The numerical VA deliberately belongs to the opposite physical class.
      va   = 64'h10000180;
      root = mapping(va, DATA);
      run_case("VA_MMIO_maps_DDR", command(va, 3, 0, 0, 0, 0, 8, root), -1, 1);
      run_case("LR_D", command(va, 3, 0, 0, 0, 10, 8, root));
      run_case("SC_D_success", command(va, 3, 0, 0, 64'h8899aabbccddeeff, 11, 8, root), 0);
      sc_successes++;
      run_case("LR_W_upper", command(va + 4, 2, 0, 0, 0, 10, 8, root));
      run_case("SC_W_success", command(va + 4, 2, 0, 0, 64'h7fffffff, 11, 8, root), 0);
      sc_successes++;
      for (int op = 1; op <= 9; op++) begin
        run_case("AMO_W_upper", command(va + 4, 2, 0, 0, 64'h80000001 + op, op, 8, root));
        run_case("AMO_neighbor_preserved", command(va, 2, 0, 0, 0, 0, 8, root));
      end
      run_case("dirty_private_eviction", command(DATA + 4096));
      run_case("dirty_shared_eviction_to_DDR", command(DATA + 8192));
      if (memory_writes == 0)
        `uvm_fatal("DDR_WRITEBACK", "Dirty data never reached the explicit DDR dictionary")
      run_case("DDR_writeback_readback", command(DATA));
      va   = 64'h80000010;
      root = mapping(va, `VM_MMIO_BASE + 16);
      run_case("VA_DDR_maps_MMIO", command(va, 1, 0, 1, 0, 0, 8, root), -1, 1);
      run_case("MMIO_store", command(va, 1, 1, 0, 64'hf123, 0, 8, root));
      run_case("MMIO_readback", command(va, 1, 0, 1, 0, 0, 8, root));
      run_case("MMIO_backend_error", command(`VM_MMIO_BASE + 24, 3));
      run_case("DDR_backend_error", command(ERROR_DATA));
      run_case("MMIO_atomic_denied", command(`VM_MMIO_BASE, 2, 0, 0, 1, 2));
      run_case("unknown_PA", command(64'h40000000));
      run_case("short_MMIO_last_byte", command(`VM_SHORT_MMIO_BASE + 4, 0, 0, 1));
      run_case("cross_region_word", command(`VM_SHORT_MMIO_BASE + 4, 2));
      run_case("cross_region_double", command(`VM_SHORT_MMIO_BASE, 3));
      run_case("DDR_last_byte", command(`VM_DDR_BASE + `VM_DDR_BYTES - 1, 0));
      run_case("DDR_after_end", command(`VM_DDR_BASE + `VM_DDR_BYTES, 0));
      run_case("misalign_before_missing_PTE", command(1, 3, 0, 0, 0, 0, 8, 0));
      run_case("noncanonical", command(64'h4000000000, 0, 0, 0, 0, 0, 8, 0));
      va   = 64'h4180;
      root = mapping(va, DATA, 64'h43);
      run_case("LR_read_only", command(va, 3, 0, 0, 0, 10, 8, root));
      for (int op = 1; op <= 9; op++)
        run_case("AMO_needs_store_permission", command(va, 3, 0, 0, 1, op, 8, root));
      run_case("SC_needs_store_permission", command(va, 3, 0, 0, 1, 11, 8, root));
      root = mapping(va, DATA, 64'h4f);
      run_case("Svade_dirty_fault", command(va, 3, 1, 0, 1, 0, 8, root));
      root = mapping(va, DATA, 64'hf);
      run_case("Svade_A_bit_page_fault", command(va, 3, 0, 0, 0, 0, 8, root));
      root = mapping(va, DATA, 64'h59);
      run_case("SUM_denied", command(va, 3, 0, 0, 0, 0, 8, root, 1, 0, 1));
      run_case("SUM_MXR_allowed", command(va, 3, 0, 0, 0, 0, 8, root, 1, 1, 1), -1, 1);
      run_case("U_load_ignores_SUM", command(va, 3, 0, 0, 0, 0, 8, root, 0, 0, 1));
      root = mapping(va, DATA, 64'hcf);
      run_case("U_cannot_read_supervisor_page", command(va, 3, 0, 0, 0, 0, 8, root, 0));
      run_case("M_bypasses_satp", command(DATA, 3, 0, 0, 0, 0, 8, 0, 3));
      root = mapping(64'hffffffc000000180, DATA);
      run_case("negative_canonical_kernel_VA", command(64'hffffffc000000180, 3, 0, 0, 0, 0, 8, root
               ));
      root = mapping(va, DATA, 64'hcf, 1);
      run_case("PTE_backend_error", command(va, 3, 0, 0, 0, 0, 8, root));
      mmu_ref_program(mmu, `VM_MMIO_BASE, 0, 1);
      run_case("PTE_cannot_access_MMIO", command(va, 3, 0, 0, 0, 0, 8, `VM_MMIO_BASE >> 12));
      mmu_ref_program(mmu, 64'h20000000, 0, 1);
      run_case("unknown_PTE_region", command(va, 3, 0, 0, 0, 0, 8, 64'h20000));
      root = mapping(va, 64'h40000180);
      run_case("translated_unknown_PA", command(va, 3, 0, 0, 0, 0, 8, root));
      root = mapping(va, 64'h1 << `VM_CACHE_ADDR_WIDTH);
      run_case("translated_PA_overflow", command(va, 3, 0, 0, 0, 0, 8, root));
      // Instruction words share the real PTE/data cache path; Fetch owns RVC assembly.
      va   = 64'h4180;
      root = mapping(va, DATA, 64'h49);
      run_case("X_only_data_denied", command(va, 3, 0, 0, 0, 0, 8, root));
      run_case("X_only_instruction_word", command(va, 3, 0, 0, 0, 0, 8, root, 1, 0, 0, 1), -1, 1);
      run_case("U_execute_supervisor_denied", command(va, 3, 0, 0, 0, 0, 8, root, 0, 1, 1, 1));
      root = mapping(va, DATA, 64'h59);
      run_case("U_execute_user_word", command(va, 3, 0, 0, 0, 0, 8, root, 0, 0, 0, 1));
      run_case("S_execute_user_SUM_denied", command(va, 3, 0, 0, 0, 0, 8, root, 1, 1, 1, 1));
      root = mapping(va, DATA, 64'h43);
      run_case("readable_nonX_instruction_denied", command(va, 3, 0, 0, 0, 0, 8, root, 1, 0, 1, 1));
      run_case("instruction_MMIO_no_side_effect", command(
               `VM_MMIO_BASE, 3, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1));
      root = mapping(va, `VM_MMIO_BASE + 384, 64'h49);
      run_case("translated_instruction_MMIO_denied", command(va, 3, 0, 0, 0, 0, 8, root, 1, 0, 0, 1
               ));
      // Two explicit aligned requests straddle a page; this owner never splits accesses.
      root = mapping(64'h4ff8, DATA, 64'h49);
      initialize_line((DATA & ~64'hfff) + 64'hfc0, pattern);
      run_case("page_last_instruction_word", command(64'h4ff8, 3, 0, 0, 0, 0, 8, root, 1, 0, 0, 1));
      // The next PTE is explicitly initialized invalid in the same fixture line.
      mmu_ref_program(mmu, (root << 12) + 8192 + 5 * 8, 0, 0);
      run_case("page_second_instruction_PF", command(64'h5000, 3, 0, 0, 0, 0, 8, root, 1, 0, 0, 1));
      root = mapping(64'h5000, 64'h40000000, 64'h49);
      run_case("page_second_instruction_AF", command(64'h5000, 3, 0, 0, 0, 0, 8, root, 1, 0, 0, 1));
      run_case("PMA_read_only_load", command(`VM_READ_ONLY_BASE));
      run_case("PMA_read_only_store_denied", command(`VM_READ_ONLY_BASE, 3, 1, 0, 64'hbadbad));
      run_case("PMA_read_only_AMO_denied", command(`VM_READ_ONLY_BASE, 2, 0, 0, 1, 2));
      run_case("PMA_cached_nonexecutable", command(
               `VM_READ_ONLY_BASE, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1));
      run_case("PMA_write_only_read_denied", command(`VM_WRITE_ONLY_BASE));
      // Authorization uses a policy captured for this accepted command. Denials
      // reuse the MMU backend-error result contract without modifying DDR data.
      deny_final = 1;
      run_case("authorization_deny_load", command(DATA), -1, 1);
      run_case("authorization_deny_store", command(DATA, 3, 1, 0, 64'hdeadbeef), -1, 1);
      run_case("authorization_deny_AMO", command(DATA, 2, 0, 0, 1, 2), -1, 1);
      run_case("authorization_deny_execute", command(DATA, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1), -1, 1);
      run_case("authorization_deny_MMIO", command(`VM_MMIO_BASE, 3, 1, 0, 64'hbadbad), -1, 1);
      deny_final = 0;
      run_case("denied_writes_preserve_DDR", command(DATA));
      run_case("denied_write_preserves_MMIO", command(`VM_MMIO_BASE));
      for (int denied_level = 0; denied_level < 3; denied_level++) begin
        va = 64'h4180;
        root = mapping(va, DATA);
        denied_pte=(root<<12)+(2-denied_level)*4096+((va>>(12+9*denied_level))&511)*8;
        deny_pte = 1;
        run_case("authorization_deny_PTE", command(va, 3, 0, 0, 0, 0, 8, root), -1, 1);
        deny_pte = 0;
        run_case("PTE_authorization_reenabled", command(va, 3, 0, 0, 0, 0, 8, root));
      end
      if (auth_denials != 8 || !auth_stalls || auth_pending)
        `uvm_fatal("AUTH_SCENARIOS", "Authorization denials/stalls/drain missing")
      `uvm_info(
          "AUTHORIZATION",
          $sformatf(
              "queries=%0d denies=%0d queryStalls=%0d; PTE/final fields and no-before-grant backend checked",
              auth_queries, auth_denials, auth_stalls), UVM_LOW)
      // No reservation timing model: require a real clean Evict of the LR line
      // during the subsequent walk and independently check SC's architectural 1.
      va   = 64'h4000;
      root = mapping(va, EVICT_DATA);
      run_case("LR_before_PTE_eviction", command(EVICT_DATA, 3, 0, 0, 0, 10));
      prior = evictions.exists(EVICT_DATA) ? evictions[EVICT_DATA] : 0;
      run_case("SC_after_real_PTE_eviction", command(va, 3, 0, 0, 64'h1234, 11, 8, root), 1);
      if (!evictions.exists(EVICT_DATA) || evictions[EVICT_DATA] <= prior)
        `uvm_fatal("SC_EVICTION", "No real cache eviction witnessed between LR and failed SC")
      sc_evictions++;
      run_case("failed_SC_preserves_data", command(EVICT_DATA));
      initialize_line(`VM_NORMAL_RAM_BASE, 512'hfedcba98765432100123456789abcdef);
      run_case("normal_RAM_load", command(`VM_NORMAL_RAM_BASE));
      run_case("normal_RAM_execute", command(`VM_NORMAL_RAM_BASE, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1));
      run_case("normal_RAM_store", command(`VM_NORMAL_RAM_BASE + 8, 3, 1, 0, 64'h8123456789abcdef));
      run_case("normal_RAM_store_readback", command(`VM_NORMAL_RAM_BASE + 8));
      next_table = `VM_NORMAL_RAM_BASE + 64'h10000;
      va = 64'h12345000;
      root = mapping(va, `VM_NORMAL_RAM_BASE);
      run_case("uncached_PTE_uncached_data", command(va, 3, 0, 0, 0, 0, 8, root), -1, 1);
      run_case("uncached_PTE_execute", command(va, 3, 0, 0, 0, 0, 8, root, 1, 0, 0, 1));
      va   = 64'h12345180;
      root = mapping(va, DATA);
      run_case("uncached_PTE_cached_data", command(va, 3, 0, 0, 0, 0, 8, root));
      root = mapping(va, DATA, 8'h0f);
      run_case("uncached_PTE_missing_AD_fault", command(va, 3, 0, 0, 0, 0, 8, root));
      root = mapping(va, DATA);
      begin
        u64 address = (root << 12) + ((va >> 30) & 511) * 8;
        program_pte(address, read_arch(address, 8, 3), 1);
      end
      run_case("uncached_PTE_backend_error", command(va, 3, 0, 0, 0, 0, 8, root));
      root = mapping(va, DATA);
      denied_pte = (root << 12) + ((va >> 30) & 511) * 8;
      deny_pte = 1;
      run_case("uncached_PTE_authorization_denied", command(va, 3, 0, 0, 0, 0, 8, root), -1, 1);
      deny_pte = 0;
      line_error[`VM_NORMAL_RAM_BASE] = 1;
      run_case("normal_RAM_execute_error", command(
               `VM_NORMAL_RAM_BASE, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1));
      run_case("normal_RAM_load_error", command(`VM_NORMAL_RAM_BASE));
      if (uncached_pte_reads < 10)
        `uvm_fatal("UNCACHED_PTE", "Uncached page walk cases were not exercised")
      if(expected!=accepted||expected!=returned||!pte_reads||!data_accesses||!uncached_accesses||!req_stalls||!resp_stalls||!mem_stalls||!uncached_stalls||!page_faults||!access_faults||!misalignments||!sc_successes||!sc_evictions)
        `uvm_fatal("SCENARIOS", "Virtual LSU scenarios not all exercised")
      `uvm_info(
          "CHECKS",
          $sformatf(
              "virtual LSU checked=%0d realCachePTE=%0d dataCache=%0d MMIO=%0d DDRreads/writes=%0d/%0d PF/AF/misaligned=%0d/%0d/%0d SCsuccess/evicted=%0d/%0d stalls(req/resp/DDR/MMIO)=%0d/%0d/%0d/%0d; all queues drained",
              expected, pte_reads, data_accesses, uncached_accesses, memory_reads, memory_writes,
              page_faults, access_faults, misalignments, sc_successes, sc_evictions, req_stalls,
              resp_stalls, mem_stalls, uncached_stalls), UVM_LOW)
    endtask
    function void final_phase(uvm_phase phase);
      mmu_ref_destroy(mmu);
    endfunction
  endclass
endpackage
