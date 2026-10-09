class protocol_test extends ip_test;
  `uvm_component_utils(protocol_test)
  virtual cpu_control_if control;
  virtual stream_if #(`CPU_REQ_WIDTH) request;
  virtual stream_if #(`CPU_RESP_WIDTH) response;
  virtual stream_if #(`CPU_CACHE_WIDTH) cache_request;
  virtual stream_if #(`CPU_CRESULT_WIDTH) cache_response;
  virtual stream_if #(`CPU_UNCACHED_WIDTH) uncached_request;
  virtual stream_if #(`CPU_URESULT_WIDTH) uncached_response;
  int checked = 0, cache_checked = 0, uncached_checked = 0, faults_checked = 0, cancelled = 0;
  function new(string name, uvm_component parent);
    super.new(name, parent);
    timeout = 200us;
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual cpu_control_if)::get(
            this, "", "control", control
        ) || !uvm_config_db#(virtual stream_if #(`CPU_REQ_WIDTH))::get(
            this, "", "request", request
        ) || !uvm_config_db#(virtual stream_if #(`CPU_RESP_WIDTH))::get(
            this, "", "response", response
        ) || !uvm_config_db#(virtual stream_if #(`CPU_CACHE_WIDTH))::get(
            this, "", "cache_request", cache_request
        ) || !uvm_config_db#(virtual stream_if #(`CPU_CRESULT_WIDTH))::get(
            this, "", "cache_response", cache_response
        ) || !uvm_config_db#(virtual stream_if #(`CPU_UNCACHED_WIDTH))::get(
            this, "", "uncached_request", uncached_request
        ) || !uvm_config_db#(virtual stream_if #(`CPU_URESULT_WIDTH))::get(
            this, "", "uncached_response", uncached_response
        ))
      `uvm_fatal("VIF", "CPU memory interfaces missing")
  endfunction
  task transact(longint unsigned addr, int size, bit write, bit signed_load, longint unsigned data,
                int atomic_op, bit cacheable, longint unsigned raw = 64'h80ff7f0181fe8000,
                bit error = 0, bit normal_ram = 0, bit device = 0);
    req_t packet = '0;
    int unsigned target, misaligned, access_fault, bus_mask, atomic_word;
    longint unsigned bus_addr, bus_data, expected_data;
    bit [`CPU_RESP_WIDTH-1:0] expected_response;
    int tag = checked & ((1 << `CPU_TAG_BITS) - 1);
    cpu_mem_ref_prepare(addr, size, write, data, atomic_op, cacheable,
                        !device && (cacheable || normal_ram), `CPU_PHYSICAL_BITS, target,
                        misaligned, access_fault, bus_addr, bus_data, bus_mask, atomic_word);
    expected_data = target == 0 ? 0 :
        cpu_mem_ref_result(addr, size, write, signed_load, atomic_op, cacheable, raw, error);
    expected_response = '0;
    expected_response[`CPUF(RESP, TAG)] = tag;
    expected_response[`CPUF(RESP, DATA)] = expected_data;
    expected_response[`CPUF(RESP, MISALIGNED)] = misaligned;
    expected_response[`CPUF(RESP, ACCESSFAULT)] = access_fault || (target != 0 && error);
    packet[`CPUF(REQ, ADDR)] = addr;
    packet[`CPUF(REQ, TAG)] = tag;
    packet[`CPUF(REQ, SIZE)] = size;
    packet[`CPUF(REQ, WRITE)] = write;
    packet[`CPUF(REQ, SIGNED)] = signed_load;
    packet[`CPUF(REQ, DATA)] = data;
    packet[`CPUF(REQ, ATOMIC)] = atomic_op;
    packet[`CPUF(REQ, CACHEABLE)] = cacheable;
    packet[`CPUF(REQ, NORMAL)] = !device && (cacheable || normal_ram);
    @(control.cb);
    response.ready <= 0;
    cache_request.ready <= 0;
    uncached_request.ready <= 0;
    request.bits <= packet;
    request.valid <= 1;
    do @(request.sample); while (!request.sample.ready);
    request.valid <= 0;
    if (target == 1) begin
      do @(cache_request.sample); while (!cache_request.sample.valid);
      if (cache_request.sample.bits[
          `CPUF(CACHE, ADDR)
          ] !== bus_addr || cache_request.sample.bits[
          `CPUF(CACHE, DATA)
          ] !== bus_data || cache_request.sample.bits[
          `CPUF(CACHE, MASK)
          ] !== bus_mask || cache_request.sample.bits[
          `CPUF(CACHE, WRITE)
          ] !== write || cache_request.sample.bits[
          `CPUF(CACHE, ATOMIC)
          ] !== atomic_op || cache_request.sample.bits[
          `CPUF(CACHE, ATOMICWORD)
          ] !== atomic_word || uncached_request.valid)
        `uvm_fatal("CACHE", $sformatf(
                   "Cache translation mismatch addr=%h size=%0d atomic=%0d got=%h",
                   addr,
                   size,
                   atomic_op,
                   cache_request.sample.bits
                   ))
      repeat (3) @(control.cb);
      cache_request.ready <= 1;
      @(cache_request.sample);
      cache_request.ready <= 0;
      repeat (2) @(control.cb);
      cache_response.bits <= '0;
      cache_response.bits[`CPUF(CRESULT, DATA)] <= raw;
      cache_response.bits[`CPUF(CRESULT, ERROR)] <= error;
      cache_response.valid <= 1;
      do @(cache_response.sample); while (!cache_response.sample.ready);
      cache_response.valid <= 0;
      cache_checked++;
    end else if (target == 2) begin
      cache_response.bits[`CPUF(CRESULT, DATA)] <= ~raw;
      do @(uncached_request.sample); while (!uncached_request.sample.valid);
      if (uncached_request.sample.bits[
          `CPUF(UNCACHED, ADDR)
          ] !== addr || uncached_request.sample.bits[
          `CPUF(UNCACHED, SIZE)
          ] !== size || uncached_request.sample.bits[
          `CPUF(UNCACHED, WRITE)
          ] !== write || uncached_request.sample.bits[
          `CPUF(UNCACHED, DATA)
          ] !== data || uncached_request.sample.bits[
          `CPUF(UNCACHED, TAG)
          ] !== tag || cache_request.valid)
        `uvm_fatal("MMIO", "Uncached access expanded or changed its address/size/data/tag")
      repeat (3) @(control.cb);
      uncached_request.ready <= 1;
      @(uncached_request.sample);
      uncached_request.ready <= 0;
      repeat (2) @(control.cb);
      uncached_response.bits <= '0;
`ifdef CPU_BAD_TAG
      uncached_response.bits[`CPUF(URESULT, TAG)] <= tag ^ 1;
`else
      uncached_response.bits[`CPUF(URESULT, TAG)] <= tag;
`endif
      uncached_response.bits[`CPUF(URESULT, DATA)] <= raw;
      uncached_response.bits[`CPUF(URESULT, ERROR)] <= error;
      uncached_response.valid <= 1;
      do @(uncached_response.sample); while (!uncached_response.sample.ready);
      uncached_response.valid <= 0;
      uncached_checked++;
    end else begin
      repeat (3) begin
        @(control.cb);
        if (cache_request.valid || uncached_request.valid)
          `uvm_fatal("FAULT", "Faulting request reached an external memory port")
      end
      faults_checked++;
    end
    do @(response.sample); while (!response.sample.valid);
    if (response.sample.bits !== expected_response)
      `uvm_fatal("RESULT", $sformatf(
                 "CPU result mismatch addr=%h size=%0d atomic=%0d got=%h expected=%h",
                 addr,
                 size,
                 atomic_op,
                 response.sample.bits,
                 expected_response
                 ))
    repeat (3) begin
      @(control.cb);
      if (request.ready || response.bits !== expected_response || !response.valid ||
        cache_request.valid || uncached_request.valid)
        `uvm_fatal("SINGLE", "Outstanding CPU response changed or accepted another operation")
    end
    response.ready <= 1;
    @(response.sample);
    response.ready <= 0;
    checked++;
  endtask
  // Reset jointly cancels the CPU operation and its memory response producer; stores are not rolled back.
  task cancel_at(int phase_index, bit cacheable);
    req_t packet = '0;
    packet[`CPUF(REQ, ADDR)] = 64'h5000;
    packet[`CPUF(REQ, SIZE)] = 3;
    packet[`CPUF(REQ, TAG)] = 63;
    packet[`CPUF(REQ, CACHEABLE)] = cacheable;
    packet[`CPUF(REQ, NORMAL)] = cacheable;
    @(control.cb);
    request.bits <= packet;
    request.valid <= 1;
    cache_request.ready <= 0;
    uncached_request.ready <= 0;
    response.ready <= 0;
    do @(request.sample); while (!request.sample.ready);
    request.valid <= 0;
    if (cacheable) begin
      do @(cache_request.sample); while (!cache_request.sample.valid);
      if (phase_index != 0) begin
        cache_request.ready <= 1;
        @(cache_request.sample);
        cache_request.ready <= 0;
        if (phase_index == 2) begin
          cache_response.bits  <= '0;
          cache_response.valid <= 1;
          do @(cache_response.sample); while (!cache_response.sample.ready);
          cache_response.valid <= 0;
          do @(response.sample); while (!response.sample.valid);
        end
      end
    end else begin
      do @(uncached_request.sample); while (!uncached_request.sample.valid);
      if (phase_index != 0) begin
        uncached_request.ready <= 1;
        @(uncached_request.sample);
        uncached_request.ready <= 0;
        if (phase_index == 2) begin
          uncached_response.bits <= '0;
          uncached_response.bits[`CPUF(URESULT, TAG)] <= 63;
          uncached_response.valid <= 1;
          do @(uncached_response.sample); while (!uncached_response.sample.ready);
          uncached_response.valid <= 0;
          do @(response.sample); while (!response.sample.valid);
        end
      end
    end
    control.cb.reset <= 1;
    cache_response.valid <= 0;
    // Payload and VALID are ignored during joint reset, including an unrelated tag.
    uncached_response.bits <= '0;
    uncached_response.valid <= !cacheable && phase_index == 1;
    repeat (2) @(control.cb);
    uncached_response.valid <= 0;
    control.cb.reset <= 0;
    repeat (2) @(control.cb);
    if (response.valid || cache_request.valid || uncached_request.valid || !request.ready)
      `uvm_fatal("RESET", "Reset retained a cancelled transaction")
    cancelled++;
  endtask
  task execute();
    control.reset = 1;
    request.valid = 1;
    request.bits = '1;
    response.ready = 0;
    cache_request.ready = 0;
    uncached_request.ready = 0;
    cache_response.valid = 0;
    cache_response.bits = '0;
    uncached_response.valid = 0;
    uncached_response.bits = '0;
    repeat (4) @(control.cb);
    request.valid <= 0;
    control.cb.reset <= 0;
`ifdef CPU_BAD_CASE
    begin
      req_t bad = '0;
      bad[`CPUF(REQ, SIZE)] = `CPU_BAD_CASE == 1 ? 4 : (`CPU_BAD_CASE == 4 ? 1 : 3);
      bad[`CPUF(REQ, ATOMIC)] = `CPU_BAD_CASE == 2 ? 12 : (`CPU_BAD_CASE == 3 ? 15 : 1);
      bad[`CPUF(REQ, WRITE)] = `CPU_BAD_CASE == 5;
      bad[`CPUF(REQ, CACHEABLE)] = 1;
      @(control.cb);
      request.bits  <= bad;
      request.valid <= 1;
      repeat (10) @(control.cb);
      `uvm_fatal("NO_ASSERT", "Invalid CPU memory contract was accepted")
    end
`elsif CPU_BAD_TAG
    transact(64'h60020000, 0, 0, 0, 0, 0, 0);
    `uvm_fatal("NO_ASSERT", "Incorrect uncached response tag was accepted")
`else
    for (int phase_index = 0; phase_index < 3; phase_index++)
    for (int cacheable = 0; cacheable < 2; cacheable++) cancel_at(phase_index, cacheable);
    for (int cacheable = 0; cacheable < 2; cacheable++)
    for (int size = 0; size < 4; size++)
    for (int offset = 0; offset < 8; offset++)
    for (int sign_bit = 0; sign_bit < 2; sign_bit++) begin
      transact(64'h60020000 + offset, size, 0, sign_bit, 0, 0, cacheable);
      transact(64'h60020000 + offset, size, 1, 0, 64'hfedcba9876543210, 0, cacheable);
    end
    // Every aligned upper/lower Word atomic lane, including sign-expanded results and SC status.
    for (int op = 1; op <= 11; op++)
    for (int size = 2; size < 4; size++)
    for (int offset = 0; offset < 8; offset += 4) begin
      longint unsigned raw = op == 11 ? 1 : (size == 2 ? 64'hffffffff80000081 : 64'h8123456789abcdef);
      transact(64'h1000 + offset, size, 0, 0, 64'h8877665544332211, op, 1, raw);
      transact(64'h1000 + offset, size, 0, 1, 64'h1122334455667788, op, 1, raw, 1);
      transact(64'h1000 + offset, size, 0, 0, 1, op, 0, raw);
    end
    for (int size = 2; size < 4; size++)
    for (int offset = 1; offset < 8; offset++)
    if (offset % (1 << size) != 0) transact(64'h1000 + offset, size, 0, 0, 1, 1, 1);
    for (int size = 0; size < 4; size++) begin
      transact(64'h2000, size, 0, 1, 0, 0, 1, '1);
      transact(64'h2000, size, 0, 0, 0, 0, 1, 0);
      transact(64'h2000, size, 0, 0, 0, 0, 1, '1, 1);
      transact(64'h60020004, size, 0, 1, 0, 0, 0, '1, 1);
      transact(64'h60020004, size, 1, 0, '1, 0, 0, 0, 1);
      transact(64'h60020000, size, 0, 1, 0, 0, 0, '1);
      transact(64'h60020000, size, 0, 0, 0, 0, 0, '1);
      transact(64'h60020000, size, 1, 0, '1, 0, 0, 0, 1);
    end
    for (int bit_index = 3; bit_index < 64; bit_index++) begin
      transact(64'h1 << bit_index, 3, 0, 0, 0, 0, 1);
      transact(64'h1 << bit_index, 3, 1, 0, '1, 0, 0);
    end
    transact(64'h1004, 2, 0, 1, 0, 0, 1, 0);
    transact(64'h60020004, 2, 0, 1, 0, 0, 0, 0);
    transact(64'h1000, 3, 1, 0, 0, 0, 1, 0, 1);
    transact(64'h1000, 2, 0, 0, 1, 11, 1, 0);
    transact(64'h1004, 2, 0, 0, 1, 11, 1, 0);
    transact(64'h1000, 3, 0, 0, 1, 11, 1, 0);
    transact(64'h1000, 3, 1, 0, 0, 0, 1);
    transact(64'h1000, 3, 1, 0, '1, 0, 1);
    transact((64'h1 << `CPU_PHYSICAL_BITS) - 8, 3, 0, 0, 0, 0, 1);
    transact((64'h1 << `CPU_PHYSICAL_BITS) - 1, 0, 1, 0, '1, 0, 1);
    // Noncacheable normal memory, including every atomic, faults without issuing traffic.
    for (int size = 0; size < 4; size++) begin
      transact(64'h3000, size, 0, 0, 0, 0, 0, 0, 0, 1);
      transact(64'h3000, size, 1, 0, 1, 0, 0, 0, 0, 1);
      transact(64'h3001, size, 0, 0, 0, 0, 0, 0, 0, 1);
      transact(64'h60020000, size, 0, 0, 0, 0, 1, 0, 0, 0, 1);
      transact(64'h60020000, size, 1, 0, 1, 0, 1, 0, 0, 0, 1);
      transact(64'h60020001, size, 0, 0, 0, 0, 1, 0, 0, 0, 1);
    end
    for (int op = 1; op <= 11; op++)
    for (int size = 2; size <= 3; size++) begin
      transact(64'h3000 + (size == 2 ? 4 : 0), size, 0, 0, 64'h76543210, op, 0,
               op == 11 ? 0 : 64'hffffffff81234567, 0, 1);
      transact(64'h3000, size, 0, 0, 1, op, 0, 64'hdeadbeef12345678, 1, 1);
    end
    transact(64'h3004, 2, 0, 0, 1, 11, 0, 1, 0, 1);
    // Reset ignores even VALID=1 garbage payload; no transaction may escape reset.
    control.cb.reset <= 1;
    request.valid <= 1;
    request.bits <= '1;
    repeat (2) @(control.cb);
    request.valid <= 0;
    request.bits <= '0;
    control.cb.reset <= 0;
    repeat (2) @(control.cb);
    if (response.valid || cache_request.valid || uncached_request.valid || !request.ready)
      `uvm_fatal("RESET_PAYLOAD", "Reset payload escaped as a transaction")
    // VALID=0 payload is unconstrained; it must not create an operation.
    response.ready <= 1;
    cache_request.ready <= 1;
    uncached_request.ready <= 1;
    request.bits <= '1;
    cache_response.bits <= '1;
    uncached_response.bits <= '1;
    repeat (2) @(control.cb);
    request.bits <= '0;
    cache_response.bits <= '0;
    uncached_response.bits <= '0;
    repeat (2) @(control.cb);
    if (response.valid || cache_request.valid || uncached_request.valid || !request.ready)
      `uvm_fatal("IDLE", "Idle payload generated an operation")
    `uvm_info("CPU_MEM", $sformatf(
              "Checked %0d CPU results, %0d cache requests, %0d exact uncached requests, %0d suppressed faults, %0d joint-reset cancellations",
              checked,
              cache_checked,
              uncached_checked,
              faults_checked,
              cancelled
              ), UVM_LOW)
`endif
  endtask
endclass
