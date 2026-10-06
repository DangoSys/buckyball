`include "admission_bridge_config.svh"
package admission_bridge_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  `define RF(F) `AB_RET_``F``_OFFSET +: `AB_RET_``F``_WIDTH
  `define FF(F) `AB_FAULT_``F``_OFFSET +: `AB_FAULT_``F``_WIDTH
  `define SF(F) `AB_SNAP_``F``_OFFSET +: `AB_SNAP_``F``_WIDTH
  typedef bit [`AB_SNAP_WIDTH-1:0] snapshot_t;
  class protocol_test extends ip_test;
    `uvm_component_utils(protocol_test)
    virtual admission_bridge_control_if control;
    virtual stream_if #(`AB_SNAP_WIDTH) command;
    virtual stream_if #(`AB_RET_WIDTH)  retirement;
    virtual stream_if #(`AB_NPU_WIDTH)  npu;
    typedef struct {
      snapshot_t snapshot;
      int rob;
      bit pending;
      bit [3:0] error;
      bit [63:0] address;
    } context_t;
    context_t contexts[int];
    int accepted = 0, notices = 0, cancelled = 0, peak = 0, lookup_hits = 0, lookup_misses = 0;
    bit late_fault = 0;
    int bound_faults = 0, repeated_faults = 0, unbound_faults = 0, error_notices = 0;
    int multi_retire = 0, boot_ignored = 0, nonrob = 0, stalls = 0, cycle = 0;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      timeout = 100us;
    endfunction
    function void verify_contract(bit ok, string message);
      if (!ok) `uvm_fatal("ADMISSION_BRIDGE", message)
    endfunction
    function snapshot_t packet(int tag, int funct, int seed);
      snapshot_t value;
      // All CSR/PMP/instruction bits participate in independent exact comparisons.
      for (int i = 0; i < `AB_SNAP_WIDTH; i++)
      value[i] = ((i * 17 + seed * 7) % (seed % 19 + 3)) % 2;
      value[`SF(TAG)] = tag;
      value[`SF(INSTRUCTION_FUNCT)] = funct;
      return value;
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      void'(uvm_config_db#(bit)::get(this, "", "late_fault", late_fault));
      verify_contract(uvm_config_db#(virtual admission_bridge_control_if)::get(
                      this, "", "control", control), "Missing control");
      verify_contract(uvm_config_db#(virtual stream_if #(`AB_SNAP_WIDTH))::get(
                      this, "", "command", command), "Missing command");
      verify_contract(uvm_config_db#(virtual stream_if #(`AB_RET_WIDTH))::get(
                      this, "", "retirement", retirement), "Missing retirement");
      verify_contract(uvm_config_db#(virtual stream_if #(`AB_NPU_WIDTH))::get(this, "", "npu", npu),
                      "Missing NPU");
    endfunction
    task monitor();
      forever begin
        @(control.sample);
        if (control.sample.reset) begin
          cancelled += contexts.num();
          contexts.delete();
        end else begin
          int retiring = 0;
          cycle++;
          foreach (contexts[tag])
          if (contexts[tag].rob >= 0 && control.sample.retired[contexts[tag].rob]) begin
            contexts[tag].pending = 1;
            retiring++;
          end
          if (retiring > 1) multi_retire++;
          begin
            int bound = -1;
            foreach (contexts[tag])
            if (contexts[tag].rob == control.sample.fault_bits[`FF(ROB_ID)]) bound = tag;
            verify_contract(
                control.sample.unbound_fault_valid == (control.sample.fault_valid && bound < 0),
                "Unknown/boot fault was lost or misrouted");
            if (control.sample.unbound_fault_valid) begin
              verify_contract(control.sample.unbound_fault_bits === control.sample.fault_bits,
                              "Unbound fault payload changed");
              unbound_faults++;
            end
            if (control.sample.fault_valid && bound >= 0 && control.sample.fault_bits[
                `FF(ERROR)
                ] != 0 && !late_fault) begin
              if (contexts[bound].error == 0) begin
                contexts[bound].error   = control.sample.fault_bits[`FF(ERROR)];
                contexts[bound].address = control.sample.fault_bits[`FF(ADDRESS)];
                bound_faults++;
              end else repeated_faults++;
            end
          end
          for (int port = 0; port < 3; port++) begin
            int found = -1;
            foreach (contexts[tag])
            if (!contexts[tag].pending && contexts[tag].rob == control.sample.lookup_id[port]) begin
              verify_contract(found == -1, "Reference duplicate ROB");
              found = tag;
            end
            verify_contract(control.sample.lookup_valid[port] == (found >= 0),
                            "Lookup live/retired visibility mismatch");
            if (found >= 0) begin
              verify_contract(control.sample.lookup_bits[port] === contexts[found].snapshot,
                              "Lookup lost snapshot or changed CSR/PMP context");
              lookup_hits++;
            end else lookup_misses++;
          end
          verify_contract(
              (command.sample.valid&&command.sample.ready)==(npu.sample.valid&&npu.sample.ready),
              "CPU/NPU acceptance not atomic");
          if (command.sample.valid && !command.sample.ready) stalls++;
          if (npu.sample.valid)
            verify_contract(
                npu.sample.bits === command.sample.bits[`AB_INSTRUCTION_OFFSET+:`AB_NPU_WIDTH],
                "NPU instruction differs from CPU snapshot");
          if (control.sample.allocation_valid && !(npu.sample.valid && npu.sample.ready)) begin
            verify_contract(contexts.num() == 0, "Boot overlapped external contexts");
            boot_ignored++;
          end
          // Check retirement while stalled as well as when accepted.
          if (retirement.sample.valid) begin
            int tag = retirement.sample.bits[`RF(SNAPSHOT_TAG)];
            verify_contract(contexts.exists(tag) && contexts[tag].pending,
                            "Unexpected retirement notice");
            verify_contract(
                retirement.sample.bits[`AB_RET_SNAPSHOT_OFFSET+:`AB_SNAP_WIDTH]===contexts[tag].snapshot,
                "Retirement changed full snapshot");
            verify_contract(retirement.sample.bits[`RF(ERROR)
                            ] === contexts[tag].error && retirement.sample.bits[`RF(ADDRESS)
                            ] === contexts[tag].address,
                            "Retirement lost first fault/error/full64 address");
          end
          if (command.sample.valid && command.sample.ready) begin
            int tag = command.sample.bits[`SF(TAG)];
            int funct = command.sample.bits[`SF(INSTRUCTION_FUNCT)];
            int rob = funct <= 1 ? -1 : int'(control.sample.allocation_id);
            verify_contract(!contexts.exists(tag) && contexts.num() < `AB_ENTRIES,
                            "Reused tag/context in retirement acceptance cycle");
            if (rob >= 0)
              foreach (contexts[old])
              verify_contract(contexts[old].rob != rob,
                              "ROB reused before old retirement acknowledgement");
            verify_contract(control.sample.allocation_valid == (rob >= 0),
                            "Wrong allocation ownership");
            contexts[tag] = '{command.sample.bits, rob, rob < 0, 0, 0};
            accepted++;
            if (rob < 0) nonrob++;
            if (contexts.num() > peak) peak = contexts.num();
          end
          if (retirement.sample.valid && retirement.sample.ready) begin
            int tag = retirement.sample.bits[`RF(SNAPSHOT_TAG)];
            if (contexts[tag].error != 0) error_notices++;
            contexts.delete(tag);
            notices++;
          end
        end
        @(negedge control.clock);
        for (int port = 0; port < 3; port++)
        control.lookup_id[port] = (cycle + port * 5) % `AB_ROB_ENTRIES;
      end
    endtask
    task submit(int tag, int rob, int funct = 2, int seed = 3);
      @(negedge control.clock);
      control.allocation_id = rob;
      command.bits = packet(tag, funct, seed);
      command.valid = 1;
      do @(command.sample); while (!command.sample.ready);
      @(negedge control.clock);
      command.valid = 0;
      command.bits  = ~command.bits;
    endtask
    task retire(bit [`AB_ROB_ENTRIES-1:0] mask);
      @(negedge control.clock);
      control.retired = mask;
      @(control.sample);
      @(negedge control.clock);
      control.retired = 0;
    endtask
    task fault(int rob, int error, bit [63:0] address, bit [`AB_ROB_ENTRIES-1:0] mask = 0);
      @(negedge control.clock);
      control.fault_bits = '0;
      control.fault_bits[`FF(ROB_ID)] = rob;
      control.fault_bits[`FF(ERROR)] = error;
      control.fault_bits[`FF(ADDRESS)] = address;
      control.fault_valid = 1;
      control.retired = mask;
      @(control.sample);
      @(negedge control.clock);
      control.fault_valid = 0;
      control.retired = 0;
      control.fault_bits = ~control.fault_bits;
    endtask
    task blocked(int cycles);
      repeat (cycles) begin
        @(control.sample);
        verify_contract(!command.sample.ready && !npu.sample.valid, "Blocked binding reached NPU");
      end
    endtask
    task drain();
      wait (contexts.num() == 0);
      repeat (3) @(control.sample);
      verify_contract(!retirement.sample.valid, "Notice remained after all acknowledgements");
    endtask
    task execute();
      control.fault_valid = 0;
      control.fault_bits = 0;
      control.reset = 1;
      control.boot_allocation = 0;
      control.retired = 0;
      control.allocation_id = 0;
      for (int i = 0; i < 3; i++) control.lookup_id[i] = 0;
      command.valid = 0;
      command.bits = 0;
      npu.ready = 0;
      retirement.ready = 0;
      fork
        monitor();
      join_none
      repeat (4) @(negedge control.clock);
      control.reset = 0;
      if (late_fault) begin
        npu.ready = 1;
        submit(86, 8, 2, 9);
        retire(16'h100);
        wait (retirement.valid);
        fault(8, 7, 64'hf123456789abcdef);
        repeat (4) @(control.sample);
        `uvm_fatal("MISSING_ASSERTION", "Late fault did not trigger DUT ordering assertion")
        return;
      end
      for (int i = 0; i < 3; i++) begin
        @(negedge control.clock);
        control.boot_allocation = 1;
        control.allocation_id   = i;
        @(control.sample);
      end
      @(negedge control.clock);
      control.boot_allocation = 0;
      fault(14, 7, 64'hfedcba9876543210);
      // A stable external offer must survive NPU backpressure without reservation.
      fork
        submit(0, 0, 2, 7);
        begin
          repeat (5) @(control.sample);
          verify_contract(contexts.num() == 0, "NPU stall reserved a context");
          @(negedge control.clock);
          npu.ready = 1;
        end
      join
      submit(127, 3, 3, 11);
      submit(255, 7, 4, 17);
      submit(37, 15, 5, 23);
      fault(0, 3, 64'h8123456789abcdef);
      fault(0, 4, 64'hfedcba9876543210);
      fault(7, 0, 64'hffffffffffffffff);
      fault(14, 5, 64'habcdef0123456789);
      fork
        submit(2, 9, 6, 29);
        begin
          @(negedge control.clock);
          repeat (2) @(control.sample);
          blocked(4);
          fault(3, 9, 64'hfedcba9876543210, 16'h8089);
          wait (retirement.valid);
          blocked(4);
          @(negedge control.clock);
          retirement.ready = 1;
        end
      join
      wait (contexts.num() == 1);
      repeat (18) @(control.sample);
      retire(16'h200);
      drain();
      // A duplicated caller tag remains blocked after ROB retire until notice acceptance.
      @(negedge control.clock);
      retirement.ready = 0;
      submit(42, 5, 8, 31);
      fork
        submit(42, 6, 9, 37);
        begin
          repeat (2) @(control.sample);
          blocked(4);
          retire(16'h20);
          wait (retirement.valid);
          blocked(4);
          @(negedge control.clock);
          retirement.ready = 1;
        end
      join
      repeat (18) @(control.sample);
      retire(16'h40);
      drain();
      // Old pending notices keep their ROB binding even with a different caller tag.
      @(negedge control.clock);
      retirement.ready = 0;
      submit(55, 11, 10, 41);
      retire(16'h800);
      wait (retirement.valid);
      fork
        submit(56, 11, 11, 43);
        begin
          repeat (2) @(control.sample);
          blocked(5);
          @(negedge control.clock);
          retirement.ready = 1;
        end
      join
      repeat (18) @(control.sample);
      retire(16'h800);
      drain();
      @(negedge control.clock);
      retirement.ready = 0;
      submit(73, 0, 15, 45);
      submit(70, 0, 0, 47);
      submit(71, 0, 1, 53);
      repeat (5) @(control.sample);
      @(negedge control.clock);
      retirement.ready = 1;
      retire(16'h1);
      drain();
      // Coordinated reset cancels both a held notice and another unretired binding.
      @(negedge control.clock);
      retirement.ready = 0;
      submit(80, 2, 12, 59);
      submit(81, 4, 13, 61);
      retire(16'h4);
      wait (retirement.valid);
      @(negedge control.clock);
      control.reset = 1;
      repeat (3) @(negedge control.clock);
      control.reset = 0;
      retirement.ready = 1;
      repeat (18) @(control.sample);
      submit(80, 2, 14, 67);
      repeat (18) @(control.sample);
      retire(16'h4);
      drain();
      verify_contract(
          accepted==notices+cancelled&&cancelled==2&&peak==4&&boot_ignored==3&&nonrob==2&&multi_retire>0&&lookup_hits>0&&lookup_misses>0&&stalls>0&&bound_faults==2&&repeated_faults==1&&unbound_faults==2&&error_notices==2,
          "Missing lifecycle coverage or lost notice");
      `uvm_info(
          "ADMISSION_FAULT_PASS",
          $sformatf(
              "firstFaults=%0d repeatedIgnored=%0d unboundForwarded=%0d errorNotices=%0d; same-cycle fault+retire and full64 addresses checked",
              bound_faults, repeated_faults, unbound_faults, error_notices), UVM_LOW)
      `uvm_info(
          "ADMISSION_BRIDGE_PASS",
          $sformatf(
              "accepted=%0d notices=%0d resetCancelled=%0d peak=%0d bootIgnored=%0d nonROB=%0d multiRetire=%0d lookup hit/miss=%0d/%0d stalls=%0d; complete snapshot/tag/PMP retained and all contexts drained",
              accepted, notices, cancelled, peak, boot_ignored, nonrob, multi_retire, lookup_hits,
              lookup_misses, stalls), UVM_LOW)
    endtask
  endclass
endpackage
