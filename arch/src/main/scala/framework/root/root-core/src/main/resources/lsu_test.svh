class protocol_test extends ip_test;
  `uvm_component_utils(protocol_test)
  virtual lsu_control_if control;
  virtual stream_if #(`CORE_CPU_WIDTH) cpu;
  virtual stream_if #(`CORE_RETURN_WIDTH) result;
  virtual stream_if #(`CORE_VIRTUAL_WIDTH) virtual_req;
  virtual stream_if #(`CORE_RESULT_WIDTH) virtual_resp;
  int issued = 0, checked = 0, killed = 0, rejected = 0;
  function new(string name, uvm_component parent);
    super.new(name, parent);
    timeout = 50us;
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual lsu_control_if)::get(
            this, "", "control", control
        ) || !uvm_config_db#(virtual stream_if #(`CORE_CPU_WIDTH))::get(
            this, "", "cpu", cpu
        ) || !uvm_config_db#(virtual stream_if #(`CORE_RETURN_WIDTH))::get(
            this, "", "result", result
        ) || !uvm_config_db#(virtual stream_if #(`CORE_VIRTUAL_WIDTH))::get(
            this, "", "virtual_req", virtual_req
        ) || !uvm_config_db#(virtual stream_if #(`CORE_RESULT_WIDTH))::get(
            this, "", "virtual_resp", virtual_resp
        ))
      `uvm_fatal("VIF", "LSU interfaces missing")
  endfunction
  task start(int cmd = 1, longint unsigned address = 64'h800040c0);
    bit [`CORE_CPU_WIDTH-1:0] packet = '0;
    packet[`LF(CPU, ADDR)] = address;
    packet[`LF(CPU, TAG)] = 2;
    packet[`LF(CPU, CMD)] = cmd;
    packet[`LF(CPU, SIZE)] = 3;
    packet[`LF(CPU, DPRV)] = 3;
    packet[`LF(CPU, SIGNED)] = 1;
    packet[`LF(CPU, NO_RESP)] = cmd == 1;
    packet[`LF(CPU, DATA)] = 64'hdeadbeef;  // EX req.data is not the later S1 operand.
    control.cb.pc <= 46'h80000100;
    control.cb.s1_kill <= 0;
    control.cb.s2_kill <= 0;
    control.cb.s1_data <= 64'h0123456789abcdef;
    cpu.bits <= packet;
    cpu.valid <= 1;
    do @(cpu.sample); while (!cpu.sample.ready);
    cpu.valid <= 0;
  endtask
  task no_backend();
    repeat (3) begin
      @(control.cb);
      if (virtual_req.valid || result.valid)
        `uvm_fatal("KILL", "Killed instruction had a memory/CPU side effect")
    end
    killed++;
  endtask
  task execute_operation(int cmd = 1, bit fault = 0);
    start(cmd);
    @(control.cb);
    @(control.cb);
    if (!control.cb.nack) `uvm_fatal("S2", "Original operation did not nack at its precise S2")
    do @(virtual_req.sample); while (!virtual_req.sample.valid);
    if (virtual_req.sample.bits[
        `LF(VIRTUAL, TAG)
        ] != 2 || virtual_req.sample.bits[
        `LF(VIRTUAL, VADDR)
        ] != 64'h800040c0 || virtual_req.sample.bits[
        `LF(VIRTUAL, DATA)
        ] != 64'h0123456789abcdef || virtual_req.sample.bits[
        `LF(VIRTUAL, WRITE)
        ] != (cmd == 1) || virtual_req.sample.bits[
        `LF(VIRTUAL, EXECUTE)
        ] != 0)
      `uvm_fatal("OPERAND", "S1 operand/tag/address incorrectly captured")
    repeat (3) @(control.cb);
    virtual_req.ready <= 1;
    @(virtual_req.sample);
    virtual_req.ready <= 0;
    issued++;
    repeat (4) @(control.cb);
    virtual_resp.bits <= '0;
    virtual_resp.bits[`LF(RESULT, TAG)] <= 2;
    virtual_resp.bits[`LF(RESULT, DATA)] <= cmd == 0 ? 64'h89abcdef01234567 : 0;
    virtual_resp.bits[`LF(RESULT, ACCESSFAULT)] <= fault;
    virtual_resp.valid <= 1;
    do @(virtual_resp.sample); while (!virtual_resp.sample.ready);
    virtual_resp.valid <= 0;
    @(control.cb);
    if (!control.cb.ordered || result.valid)
      `uvm_fatal("ORDER",
                 "Buffered completed operation still blocked ordered decode or returned early")
  endtask
  task reject_identity(bit wrong_pc);
    bit [`CORE_CPU_WIDTH-1:0] packet = '0;
    packet[`LF(CPU, ADDR)] = 64'h800040c0;
    packet[`LF(CPU, TAG)]  = wrong_pc ? 2 : 4;
    packet[`LF(CPU, CMD)]  = 1;
    packet[`LF(CPU, SIZE)] = 3;
    packet[`LF(CPU, DPRV)] = 3;
    control.cb.pc <= wrong_pc ? 46'h80000104 : 46'h80000100;
    cpu.bits <= packet;
    cpu.valid <= 1;
    repeat (3) begin
      @(cpu.sample);
      if (cpu.sample.ready || result.valid || virtual_req.valid)
        `uvm_fatal("IDENTITY", "A different tag/PC consumed the buffered completion")
    end
    control.cb.cancel_offer <= 1;
    cpu.valid <= 0;
    @(control.cb);
    control.cb.cancel_offer <= 0;
    rejected++;
  endtask
  task replay(int cmd = 1, bit fault = 0, bit misaligned = 0,
              longint unsigned address = 64'h800040c0);
    start(cmd, address);
    @(control.cb);
    do @(result.sample); while (!result.sample.valid);
    if (result.sample.bits[
        `LF(RETURN, TAG)
        ] != 2 || result.sample.bits[
        `LF(RETURN, CMD)
        ] != cmd || result.sample.bits[
        `LF(RETURN, ADDR)
        ] != address || result.sample.bits[
        `LF(RETURN, HAS_DATA)
        ] != (cmd == 0 && !fault && !misaligned) || control.cb.ae_ld != (cmd == 0 && fault) ||
            control.cb.ae_st != (cmd == 1 && fault) || control.cb.ma_ld != (cmd == 0 && misaligned)
            || control.cb.ma_st != (cmd == 1 && misaligned) || control.cb.nack || virtual_req.valid)
      `uvm_fatal("REPLAY", "Replay return/exception/tag was not aligned with its own S2")
    if (cmd == 0 && !fault && !misaligned && result.sample.bits[
        `LF(RETURN, DATA)
        ] != 64'h89abcdef01234567)
      `uvm_fatal("DATA", "Replay data changed")
    checked++;
    @(control.cb);
  endtask
  task execute();
    control.reset = 1;
    control.cancel_offer = 0;
    control.s1_kill = 0;
    control.s2_kill = 0;
    control.pc = 0;
    control.s1_data = 0;
    cpu.valid = 0;
    cpu.bits = '0;
    virtual_req.ready = 0;
    virtual_resp.valid = 0;
    virtual_resp.bits = '0;
    repeat (4) @(control.cb);
    control.cb.reset <= 0;
    @(control.cb);
    start();
    control.cb.s1_kill <= 1;
    no_backend();
    control.cb.s1_kill <= 0;
    start();
    @(control.cb);
    control.cb.s2_kill <= 1;
    no_backend();
    control.cb.s2_kill <= 0;
    execute_operation();
    reject_identity(0);
    reject_identity(1);
    start();
    control.cb.s1_kill <= 1;
    no_backend();
    control.cb.s1_kill <= 0;
    start();
    @(control.cb);
    control.cb.s2_kill <= 1;
    no_backend();
    control.cb.s2_kill <= 0;
    replay();
    execute_operation(0);
    replay(0);
    execute_operation(0, 1);
    replay(0, 1);
    execute_operation(1, 1);
    replay(1, 1);
    start(0, 64'h800040c1);
    @(control.cb);
    @(control.cb);
    repeat (3) @(control.cb);
    if (virtual_req.valid) `uvm_fatal("MISALIGN", "Misaligned request reached backend")
    replay(0, 0, 1, 64'h800040c1);
    if (issued != 4 || checked != 5 || killed != 4 || rejected != 2)
      `uvm_fatal("COUNTS", "LSU contract cases incomplete")
    `uvm_info(
        "LSU",
        $sformatf(
            "Checked %0d precise replay completions, %0d real backend executions, %0d original/replay S1/S2 kills, %0d rejected tag/PC offers",
            checked, issued, killed, rejected), UVM_LOW)
  endtask
endclass
