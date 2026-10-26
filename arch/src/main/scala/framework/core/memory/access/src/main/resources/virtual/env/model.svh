class model extends uvm_subscriber #(cycle_item);
  `uvm_component_utils(model)
  fixture image;
  chandle walker;
  uvm_analysis_port #(result_item) expected, actual;
  uvm_analysis_port #(uvm_sequence_item) cancelled;
  typedef struct {
    longint unsigned pa;
    int size;
    bit reading, writing, executing, pte;
    int privilege;
  } authorization;
  typedef struct {
    longint unsigned address, data;
    int mask, atomic_op, word;
    bit writing, pte, uncached;
    int tag, size;
  } transfer;
  authorization authorizations[$];
  transfer transfers[$];
  bit pending;
  covergroup behavior with function sample (
      int size,
      int privilege,
      int mode,
      int target,
      bit write,
      int atomic_op,
      bit misaligned,
      bit pf,
      bit af,
      bit denied,
      bit error
  );
    option.per_instance = 1;
    sizes: coverpoint size {bins all[] = {[0 : 3]};}
    privileges: coverpoint privilege {bins all[] = {0, 1, 3};}
    modes: coverpoint mode {bins all[] = {0, 8};}
    targets: coverpoint target {bins none = {0}; bins cached = {1}; bins uncached = {2};}
    direction: coverpoint write;
    atomics: coverpoint atomic_op {
      bins ordinary = {0}; bins lr = {10}; bins sc = {11}; bins other = {[1 : 9]};
    }
    alignment: coverpoint misaligned;
    page_faults: coverpoint pf;
    access_faults: coverpoint af;
    permission: coverpoint denied;
    backend_errors: coverpoint error;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    expected = new("expected", this);
    actual = new("actual", this);
    cancelled = new("cancelled", this);
    behavior = new();
    walker = mmu_ref_create(`VA_ADDRESS_BITS, 0);
  endfunction
  function void verify(bit ok, string message);
    evidence::checks++;
    if (!ok) `uvm_fatal("VIRTUAL_ACCESS", message)
  endfunction
  function void set_pte(longint unsigned address, logic [63:0] value, bit error = 0);
    image.ptes[address]   = value;
    image.errors[address] = error;
    mmu_ref_program(walker, address, value, error);
  endfunction
  function void predict(cycle_item t);
    longint unsigned va, pa, root, data, bus_address, bus_data, raw, pte_address;
    int size, tag, privilege, mode, atomic_op, target, misaligned, af, pf, level, mask, atomic_word;
    bit writing, executing, signed_load, read_data, write_data, region_ok, cacheable, normal;
    result_item result = result_item::type_id::create("expected");
    va = `MI(t, REQUEST_BITS_VADDR);
    size = `MI(t, REQUEST_BITS_SIZE);
    tag = `MI(t, REQUEST_BITS_TAG);
    privilege = `MI(t, REQUEST_BITS_PRIVILEGE);
    mode = `MI(t, REQUEST_BITS_SATPMODE);
    root = `MI(t, REQUEST_BITS_ROOTPPN);
    data = `MI(t, REQUEST_BITS_DATA);
    writing = `MI(t, REQUEST_BITS_WRITE);
    executing = `MI(t, REQUEST_BITS_EXECUTE);
    signed_load = `MI(t, REQUEST_BITS_SIGNED);
    atomic_op = `MI(t, REQUEST_BITS_ATOMIC);
    result.tag = tag;
    misaligned = (va & ((64'd1 << size) - 1)) != 0;
    result.misaligned = misaligned;
    target = 0;
    pf = 0;
    af = 0;
    verify(!pending && authorizations.size() == 0 && transfers.size() == 0,
           "Overlapping virtual requests");
    if (!misaligned) begin
      // The fixture denies a PTE through the authorization boundary, before cache access.
      foreach (image.ptes[address])
      mmu_ref_program(walker, address, image.ptes[address],
                      image.errors[address] || image.deny_pte);
      verify(mmu_ref_translate(
             walker,
             va,
             mode,
             root,
             privilege,
             writing || (atomic_op != 0 && atomic_op != 10),
             executing,
             `MI(t, REQUEST_BITS_SUM),
             `MI(t, REQUEST_BITS_MXR),
             pa,
             pf,
             af,
             level
             ) == 0, "Missing page table fixture");
      while (mmu_ref_peek(
          walker, pte_address
      ) != 0) begin
        authorizations.push_back('{pte_address, 3, 1, 0, 0, 1, 1});
        if (!image.deny_pte) transfers.push_back('{pte_address, 0, 0, 0, 0, 0, 1, 0, 0, 3});
        verify(mmu_ref_consume(walker, pte_address) != 0, "PTE trace consumption failed");
      end
      result.page_fault   = pf;
      result.access_fault = af;
      if (!pf && !af) begin
        longint unsigned last = pa + ((64'd1 << size) - 1);
        read_data = !executing && !writing && atomic_op != 11;
        write_data = writing || (atomic_op != 0 && atomic_op != 10);
        cacheable = pa >= 4096 && last < 1052672 || pa >= 3145728 && last < 3149824;
        normal = cacheable;
        region_ok=(pa>=4096&&last<1052672)||
(pa>=2097152&&last<2101248&&!executing&&atomic_op==0)||
(pa>=3145728&&last<3149824&&!executing&&!write_data&&atomic_op==0);
        authorizations.push_back('{pa, size, read_data, write_data, executing, 0, privilege});
        if (region_ok && !image.deny_data) begin
          mem_access_ref_prepare(pa, size, writing, data, atomic_op, cacheable, normal,
                                 `VA_ADDRESS_BITS, target, misaligned, af, bus_address, bus_data,
                                 mask, atomic_word);
          verify(target != 0 && !af && !misaligned, "Invalid physical reference plan");
          transfers.push_back('{bus_address, bus_data, mask, atomic_op, atomic_word, writing, 0,
                              target == 2, tag, size});
          raw = image.word(bus_address);
          result.data = mem_access_ref_result(pa, size, writing, signed_load, atomic_op, cacheable,
                                              raw, image.backend_error);
          result.access_fault = image.backend_error;
          if (target == 1 && !image.backend_error) result.line = image.line(bus_address);
        end else result.access_fault = 1;
      end
    end
    expected.write(result);
    pending = 1;
    behavior.sample(size, privilege, mode, target, writing, atomic_op, misaligned, pf,
                    result.access_fault, image.deny_pte || image.deny_data, image.backend_error);
  endfunction
  function void write(cycle_item t);
    if (t.reset) begin
      cancelled.write(null);
      pending = 0;
      authorizations.delete();
      transfers.delete();
      mmu_ref_reset(walker);
      return;
    end
    if (`MI(t, REQUEST_VALID) && `MO(t, REQUEST_READY)) predict(t);
    if (`MO(t, AUTHORIZATIONREQUEST_VALID)) begin
      authorization a;
      verify(authorizations.size() != 0, "Unexpected authorization request");
      a = authorizations[0];
      verify(`MO(t, AUTHORIZATIONREQUEST_BITS_PADDR) == a.pa &&
             `MO(t, AUTHORIZATIONREQUEST_BITS_SIZE) == a.size &&
             `MO(t, AUTHORIZATIONREQUEST_BITS_READ) == a.reading &&
             `MO(t, AUTHORIZATIONREQUEST_BITS_WRITE) == a.writing &&
             `MO(t, AUTHORIZATIONREQUEST_BITS_EXECUTE) == a.executing &&
             `MO(t, AUTHORIZATIONREQUEST_BITS_ISPTE) == a.pte &&
             `MO(t, AUTHORIZATIONREQUEST_BITS_PRIVILEGE) == a.privilege,
             "Authorization footprint mismatch");
      if (`MI(t, AUTHORIZATIONREQUEST_READY)) void'(authorizations.pop_front());
    end
    if (`MO(t, CACHEREQUEST_VALID) || `MO(t, UNCACHEDREQUEST_VALID)) begin
      transfer a;
      bit uncached;
      verify(transfers.size() != 0, "Speculative backend request without authorization");
      a = transfers[0];
      uncached = `MO(t, UNCACHEDREQUEST_VALID);
      verify(uncached == a.uncached, "Cache/MMIO target mismatch");
      if (uncached) begin
        verify(`MO(t, UNCACHEDREQUEST_BITS_ADDR) == a.address && `MO(t, UNCACHEDREQUEST_BITS_SIZE)
               == a.size && `MO(t, UNCACHEDREQUEST_BITS_TAG) == a.tag &&
               `MO(t, UNCACHEDREQUEST_BITS_WRITE) == a.writing && `MO(t, UNCACHEDREQUEST_BITS_DATA)
               == a.data, "Uncached request mismatch");
        if (`MI(t, UNCACHEDREQUEST_READY)) void'(transfers.pop_front());
      end else begin
        verify(`MO(t, CACHEREQUEST_BITS_ADDR) == a.address && `MO(t, CACHEREQUEST_BITS_WRITE)
               == a.writing && `MO(t, CACHEREQUEST_BITS_DATA) == a.data &&
               `MO(t, CACHEREQUEST_BITS_MASK) == a.mask && `MO(t, CACHEREQUEST_BITS_ATOMIC)
               == a.atomic_op && `MO(t, CACHEREQUEST_BITS_ATOMICWORD) == a.word,
               "Cache/PTE request mismatch");
        if (`MI(t, CACHEREQUEST_READY)) void'(transfers.pop_front());
      end
    end
    if (`MO(t, RESPONSE_VALID) && `MI(t, RESPONSE_READY)) begin
      result_item result = result_item::type_id::create("actual");
      verify(pending && authorizations.size() == 0 && transfers.size() == 0,
             "Early result before the planned accesses drained");
      result.tag = `MO(t, RESPONSE_BITS_TAG);
      result.data = `MO(t, RESPONSE_BITS_DATA);
      result.line = `MO(t, RESPONSE_BITS_LINE);
      result.misaligned = `MO(t, RESPONSE_BITS_MISALIGNED);
      result.page_fault = `MO(t, RESPONSE_BITS_PAGEFAULT);
      result.access_fault = `MO(t, RESPONSE_BITS_ACCESSFAULT);
      actual.write(result);
      pending = 0;
    end
  endfunction
  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    verify(!pending && authorizations.size() == 0 && transfers.size() == 0,
           "Virtual access did not drain");
  endfunction
  function void final_phase(uvm_phase phase);
    mmu_ref_destroy(walker);
    super.final_phase(phase);
  endfunction
endclass
