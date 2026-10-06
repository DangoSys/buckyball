class protocol_test extends ip_test;
  `uvm_component_utils(protocol_test)
  virtual core_control_if control;
  virtual stream_if #(`CORE_MEMREQ_WIDTH) mem_req;
  virtual stream_if #(`CORE_MEMRESP_WIDTH) mem_resp;
  virtual stream_if #(`CORE_UNCACHED_WIDTH) uncached_req;
  virtual stream_if #(`CORE_URESP_WIDTH) uncached_resp;
  chandle model;
  bit [`CORE_MEMRESP_WIDTH-1:0] memory_queue[$];
  bit [`CORE_URESP_WIDTH-1:0] mmio_queue[$];
  bit memory_active = 0, mmio_active = 0, exited = 0;
  int stage = 0, cycles = 0, reads = 0, writes = 0, cancellations = 0;
  longint unsigned exit_code;
`ifdef IRQ_PROFILE
  localparam int STAGES = 3;
  localparam int TRAPS = 3;
  bit [2:0] raised = 0, target_retired = 0;
  int acknowledgements = 0, memory_release = 0;
  int target_retire_count[3] = '{default: 0};
  int target_access_count[3] = '{default: 0};
`elsif SUPERVISOR_PROFILE
  localparam int STAGES = 3;
  localparam int TRAPS = `SUPERVISOR_TRAP_COUNT;
`elsif FPU_PROFILE
  localparam int STAGES = 4;
  localparam int TRAPS = 1;
`else
  localparam int STAGES = 7;
  localparam int TRAPS = 5;
`endif
  int traps = 0, retired = 0;
  bit exit_retired = 0;
  longint unsigned exit_pc;
  function new(string name, uvm_component parent);
    super.new(name, parent);
    timeout = 5ms;
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual core_control_if)::get(
            this, "", "control", control
        ) || !uvm_config_db#(virtual stream_if #(`CORE_MEMREQ_WIDTH))::get(
            this, "", "mem_req", mem_req
        ) || !uvm_config_db#(virtual stream_if #(`CORE_MEMRESP_WIDTH))::get(
            this, "", "mem_resp", mem_resp
        ) || !uvm_config_db#(virtual stream_if #(`CORE_UNCACHED_WIDTH))::get(
            this, "", "uncached_req", uncached_req
        ) || !uvm_config_db#(virtual stream_if #(`CORE_URESP_WIDTH))::get(
            this, "", "uncached_resp", uncached_resp
        ))
      `uvm_fatal("VIF", "Core verification interfaces missing")
    model = core_ref_create(`TEST_IMAGE);
  endfunction
  task drive_memory();
    forever begin
      @(posedge control.clock);
      if (control.reset) begin
        mem_req.ready <= 0;
        uncached_req.ready <= 0;
        mem_resp.valid <= 0;
        mem_resp.bits <= '0;
        uncached_resp.valid <= 0;
        uncached_resp.bits <= '0;
        memory_queue.delete();
        mmio_queue.delete();
        memory_active = 0;
        mmio_active   = 0;
      end else begin
        cycles++;
`ifdef IRQ_PROFILE
        foreach (target_access_count[i]) if (control.target_access[i]) target_access_count[i]++;
`endif
        if (control.cancelled) cancellations++;
        if (control.retired) begin
          retired++;
`ifdef IRQ_PROFILE
          if (control.retired_pc == `IRQ_LOAD_PC) begin
            target_retired[0] = 1;
            target_retire_count[0]++;
          end
          if (control.retired_pc == `IRQ_AMO_PC) begin
            target_retired[1] = 1;
            target_retire_count[1]++;
          end
          if (control.retired_pc == `IRQ_STORE_PC) begin
            target_retired[2] = 1;
            target_retire_count[2]++;
          end
`endif
          if (exited && control.retired_pc == exit_pc) exit_retired = 1;
        end
        if (control.trapped) begin
          int expected_cause = traps == 0 ? 11 : (traps % 2 == 1 ? 5 : 7);
          longint unsigned expected_value = traps == 0 ? 0 : (traps <= 2 ? 64'h20000000 : 64'h800040c0);
`ifdef IRQ_PROFILE
          expected_cause = traps == 0 ? 7 : (traps == 1 ? 3 : 11);
          expected_value = 0;
          if (traps >= 3 || !raised[traps] || !target_retired[traps] ||
              control.trap_cause != (64'h8000000000000000 | expected_cause) || control.trap_value != 0)
            `uvm_fatal("IRQ",
                       "Interrupt preceded original retirement or had an unexpected cause/value")
`elsif SUPERVISOR_PROFILE
          longint unsigned expected_pc;
          case (traps)
            0: begin
              expected_cause = `SUPERVISOR_TRAP_0_CAUSE;
              expected_value = `SUPERVISOR_TRAP_0_VALUE;
              expected_pc = `SUPERVISOR_TRAP_0_PC;
            end
            1: begin
              expected_cause = `SUPERVISOR_TRAP_1_CAUSE;
              expected_value = `SUPERVISOR_TRAP_1_VALUE;
              expected_pc = `SUPERVISOR_TRAP_1_PC;
            end
            2: begin
              expected_cause = `SUPERVISOR_TRAP_2_CAUSE;
              expected_value = `SUPERVISOR_TRAP_2_VALUE;
              expected_pc = `SUPERVISOR_TRAP_2_PC;
            end
            3: begin
              expected_cause = `SUPERVISOR_TRAP_3_CAUSE;
              expected_value = `SUPERVISOR_TRAP_3_VALUE;
              expected_pc = `SUPERVISOR_TRAP_3_PC;
            end
            4: begin
              expected_cause = `SUPERVISOR_TRAP_4_CAUSE;
              expected_value = `SUPERVISOR_TRAP_4_VALUE;
              expected_pc = `SUPERVISOR_TRAP_4_PC;
            end
            5: begin
              expected_cause = `SUPERVISOR_TRAP_5_CAUSE;
              expected_value = `SUPERVISOR_TRAP_5_VALUE;
              expected_pc = `SUPERVISOR_TRAP_5_PC;
            end
          endcase
          if (control.trap_pc != expected_pc)
            `uvm_fatal("TRAP_PC", "Supervisor trap PC differs from fixture symbol")
`elsif FPU_PROFILE
          expected_cause = 2;
          expected_value = 64'h53;
          if (control.trap_pc != `FPU_ILLEGAL_PC)
            `uvm_fatal("TRAP_PC", "FPU FS-off trap PC differs from fixture symbol")
`endif
`ifndef IRQ_PROFILE
          if (traps >= TRAPS || control.trap_cause != expected_cause || control.trap_value != expected_value)
            `uvm_fatal("TRAP", $sformatf(
                       "Unexpected precise trap count=%0d cause=%h value=%h pc=%h",
                       traps,
                       control.trap_cause,
                       control.trap_value,
                       control.trap_pc
                       ))
`endif
          traps++;
        end
`ifdef IRQ_PROFILE
        if (!raised[1] && acknowledgements == 1 && control.amo_completed) begin
          control.software_irq <= 1;
          raised[1] = 1;
          `uvm_info("IRQ_RAISE", "Software level asserted while AMO result was buffered", UVM_LOW)
        end
        if (!raised[2] && acknowledgements == 2 && control.store_completed) begin
          control.external_irq <= 1;
          raised[2] = 1;
          `uvm_info("IRQ_RAISE", "External level asserted while store completion was buffered",
                    UVM_LOW)
        end
`endif
        mem_req.ready <= cycles % 7 > 1;
        uncached_req.ready <= cycles % 7 > 1;
        if (mem_req.valid && mem_req.ready) begin
          bit [`CORE_MEMRESP_WIDTH-1:0] result = '0;
          bit [511:0] data;
          longint unsigned address = mem_req.bits[`CF(MEMREQ, ADDR)];
`ifdef IRQ_PROFILE
          if (!raised[0] && address == 64'h800040c0 && !mem_req.bits[`CF(MEMREQ, WRITE)]) begin
            if (!control.load_waiting)
              `uvm_fatal("IRQ_TIMING", "Cold data DDR request did not belong to receiving load")
            control.timer_irq <= 1;
            raised[0] = 1;
            memory_release = cycles + 20;
            `uvm_info("IRQ_RAISE",
                      "Timer level asserted with cold-load DDR response held for 20 cycles",
                      UVM_LOW)
          end
`endif
          result[`CF(MEMRESP, ID)] = mem_req.bits[`CF(MEMREQ, ID)];
          if (mem_req.bits[`CF(MEMREQ, WRITE)]) begin
            core_ref_write(model, address, mem_req.bits[`CF(MEMREQ, DATA)], mem_req.bits[
                           `CF(MEMREQ, MASK)]);
            writes++;
          end else begin
            core_ref_read(model, address, data);
            result[`CF(MEMRESP, DATA)] = data;
            reads++;
          end
          memory_queue.push_back(result);
        end
        if (uncached_req.valid && uncached_req.ready) begin
          bit [`CORE_URESP_WIDTH-1:0] result = '0;
          longint unsigned address = uncached_req.bits[`CF(UNCACHED, ADDR)];
          longint unsigned value = uncached_req.bits[`CF(UNCACHED, DATA)];
          bit bad_mmio, mmio_handled = 0;
          result[`CF(URESP, TAG)] = uncached_req.bits[`CF(UNCACHED, TAG)];
          bad_mmio = !uncached_req.bits[
          `CF(UNCACHED, WRITE)
          ] || uncached_req.bits[
          `CF(UNCACHED, SIZE)
          ] != 3 || (address != 64'h10000000 && address != 64'h10000008);
`ifdef IRQ_PROFILE
          if (uncached_req.bits[
              `CF(UNCACHED, WRITE)
              ] && uncached_req.bits[
              `CF(UNCACHED, SIZE)
              ] == 3 && address == 64'h10000010)
            bad_mmio = 0;
`endif
          if (bad_mmio) `uvm_fatal("MMIO", "Unexpected or expanded Core MMIO operation")
`ifdef IRQ_PROFILE
          if (address == 64'h10000010) begin
            longint unsigned expected = acknowledgements == 0 ? 64'h80 : (acknowledgements == 1 ? 64'h8 : 64'h800);
            if (acknowledgements >= 3 || traps != acknowledgements + 1 || !raised[acknowledgements] || value != expected)
              `uvm_fatal("IRQ_ACK", "IRQ ACK did not match its pending level/cause")
            case (acknowledgements)
              0: control.timer_irq <= 0;
              1: control.software_irq <= 0;
              2: control.external_irq <= 0;
            endcase
            acknowledgements++;
            mmio_handled = 1;
          end
`endif
          if (!mmio_handled && address == 64'h10000008) begin
            if (value != stage + 1 || value > STAGES)
              `uvm_fatal("STAGE", "Firmware stage signature out of order")
            stage = value;
            `uvm_info("STAGE", $sformatf("Completed firmware stage %0d", stage), UVM_LOW)
          end else if (!mmio_handled) begin
            exit_code = value;
            // The final exit SD has rd/tag zero; observe its later precise architectural retirement.
            exit_pc   = `CORE_EXIT_PC;
            if (value != 0 || stage != STAGES)
              `uvm_fatal("FIRMWARE", $sformatf("Firmware failed: stage=%0d exit=%h", stage, value))
            exited = 1;
          end
          mmio_queue.push_back(result);
        end
        if (memory_active && mem_resp.ready) memory_active = 0;
        if (mmio_active && uncached_resp.ready) mmio_active = 0;
        begin
          bit release_mem;
          release_mem = !memory_active && memory_queue.size() != 0 && cycles % 5 == 0;
`ifdef IRQ_PROFILE
          release_mem = release_mem && cycles >= memory_release;
`endif
          if (release_mem) begin
            mem_resp.bits <= memory_queue.pop_front();
            memory_active = 1;
          end
        end
        if (!mmio_active && mmio_queue.size() != 0 && cycles % 5 == 0) begin
          uncached_resp.bits <= mmio_queue.pop_front();
          mmio_active = 1;
        end
        mem_resp.valid <= memory_active;
        uncached_resp.valid <= mmio_active;
      end
    end
  endtask
  task execute();
    control.reset = 1;
    control.timer_irq = 0;
    control.software_irq = 0;
    control.external_irq = 0;
    mem_req.ready = 0;
    uncached_req.ready = 0;
    mem_resp.valid = 0;
    mem_resp.bits = '0;
    uncached_resp.valid = 0;
    uncached_resp.bits = '0;
    repeat (5) @(control.cb);
    control.cb.reset <= 0;
    fork
      drive_memory();
    join_none
    wait (exited && exit_retired);
    if (traps != TRAPS || retired < 100)
      `uvm_fatal("COMPLETION", "Firmware ended without expected architectural execution/traps")
`ifdef IRQ_PROFILE
    if (raised != 3'b111 || target_retired != 3'b111 || acknowledgements != 3 ||
        control.timer_irq || control.software_irq || control.external_irq)
      `uvm_fatal("IRQ_COMPLETION",
                 "IRQ gate finished without all held levels, retirements and acknowledgements")
    foreach (target_access_count[i])
    if (target_access_count[i] != 1 || target_retire_count[i] != 1)
      `uvm_fatal("IRQ_ONCE", $sformatf(
                 "Target %0d executed %0d cache demands and retired %0d times",
                 i,
                 target_access_count[i],
                 target_retire_count[i]
                 ))
    `uvm_info("IRQ_CHECKED",
              "All three deferred targets executed and retired exactly once; three held-level ACKs drained",
              UVM_LOW)
`endif
    do
      @(control.cb);
    while (memory_queue.size() != 0 || mmio_queue.size() != 0 || memory_active || mmio_active);
    `uvm_info(
        "CORE",
        $sformatf(
            "Firmware checked %0d stages, exit=0; cycles=%0d memory reads=%0d writes=%0d S1/S2 cancellations=%0d precise traps=%0d retired=%0d",
            STAGES, cycles, reads, writes, cancellations, traps, retired), UVM_LOW)
    control.cb.reset <= 1;
    repeat (3) @(control.cb);
  endtask
  function void final_phase(uvm_phase phase);
    core_ref_destroy(model);
    super.final_phase(phase);
  endfunction
endclass
