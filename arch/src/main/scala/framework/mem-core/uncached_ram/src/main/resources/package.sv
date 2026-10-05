`include "ram_config.svh"
`ifdef RAM_DDR
`include "ram_ddr_config.svh"
`endif
package ram_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  `define F(K, V) `RAM_``K``_``V``_OFFSET +: `RAM_``K``_``V``_WIDTH
  import "DPI-C" function chandle ram_ref_create();
  import "DPI-C" function void ram_ref_destroy(input chandle p);
  import "DPI-C" function void ram_ref_reset(input chandle p);
  import "DPI-C" function void ram_ref_program(
    input chandle p,
    input longint unsigned address,
    input bit [511:0] data
  );
  import "DPI-C" function void ram_ref_read(
    input chandle p,
    input longint unsigned address,
    output bit [511:0] data
  );
  import "DPI-C" function void ram_ref_line_write(
    input chandle p,
    input longint unsigned address,
    input bit [511:0] data,
    input bit [63:0] mask,
    input int unsigned failed
  );
  import "DPI-C" function int unsigned ram_ref_cpu(
    input chandle p,
    input int unsigned cpu,
    input longint unsigned address,
    input int unsigned size,
    input int unsigned write,
    input int unsigned op,
    input longint unsigned operand,
    input int unsigned fault,
    output longint unsigned result
  );
  typedef longint unsigned u64;
  class result_item extends uvm_sequence_item;
    bit [511:0] data;
    bit error;
    int source, tag;
    `uvm_object_utils_begin(result_item)
      `uvm_field_int(data, UVM_DEFAULT)
      `uvm_field_int(error, UVM_DEFAULT)
      `uvm_field_int(source, UVM_DEFAULT)
      `uvm_field_int(tag, UVM_DEFAULT)
    `uvm_object_utils_end
    function new(string name = "result_item");
      super.new(name);
    endfunction
  endclass
  class protocol_test extends ip_test;
    `uvm_component_utils(protocol_test)
    virtual ram_control_if control;
    virtual stream_if #(`RAM_CPU_WIDTH) cpu[2];
    virtual stream_if #(`RAM_RESULT_WIDTH) result[2];
    virtual stream_if #(`RAM_LINE_WIDTH) line[2], memory[2];
    virtual stream_if #(`RAM_REPLY_WIDTH) reply[2], returned[2];
    typedef struct {
      int source, tag, op, size, starts;
      u64 address;
      bit write, done;
      bit [63:0] mask;
      int fault;
    } command_t;
    typedef struct {
      bit [`RAM_REPLY_WIDTH-1:0] bits;
      int due, key;
      bit terminal;
    } pending_t;
    pending_t inflight[int];
    command_t commands[int];
    pending_t pending0[$], pending1[$], active[2];
    bit busy[2] = '{0, 0};
    int faults[u64];
    bit hold_memory = 0, hold_cpu = 0;
    u64 held_line = 0;
    chandle oracle, ddr;
    keyed_scoreboard #(result_item) scoreboard;
    int cancelled = 0, parallel_progress = 0;
    int
        cycle = 0,
        accepted = 0,
        checked = 0,
        peak = 0,
        stalls = 0,
        backend_reads = 0,
        backend_writes = 0,
        sc_success = 0,
        sc_failure = 0,
        errors = 0;
    function new(string n, uvm_component p);
      super.new(n, p);
      timeout = 1ms;
    endfunction
    function void ck(bit ok, string msg);
      if (!ok) `uvm_fatal("RAM_CONTRACT", msg)
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      oracle = ram_ref_create();
      ddr = ram_ref_create();
      scoreboard = keyed_scoreboard#(result_item)::type_id::create("scoreboard", this);
      ck(uvm_config_db#(virtual ram_control_if)::get(this, "", "control", control),
         "Missing control");
      for (int i = 0; i < 2; i++) begin
        ck(uvm_config_db#(virtual stream_if #(`RAM_CPU_WIDTH))::get(
           this, "", $sformatf("cpu%0d", i), cpu[i]), "Missing CPU");
        ck(uvm_config_db#(virtual stream_if #(`RAM_RESULT_WIDTH))::get(
           this, "", $sformatf("result%0d", i), result[i]), "Missing result");
        ck(uvm_config_db#(virtual stream_if #(`RAM_LINE_WIDTH))::get(
           this, "", $sformatf("line%0d", i), line[i]), "Missing line");
        ck(uvm_config_db#(virtual stream_if #(`RAM_REPLY_WIDTH))::get(
           this, "", $sformatf("reply%0d", i), reply[i]), "Missing reply");
        ck(uvm_config_db#(virtual stream_if #(`RAM_LINE_WIDTH))::get(
           this, "", $sformatf("memory%0d", i), memory[i]), "Missing memory");
        ck(uvm_config_db#(virtual stream_if #(`RAM_REPLY_WIDTH))::get(
           this, "", $sformatf("returned%0d", i), returned[i]), "Missing returned");
      end
`ifdef RAM_DDR
      axi_build();
`endif
    endfunction
    function bit [511:0] pattern(int seed);
      bit [511:0] value;
      for (int i = 0; i < 8; i++)
      value[i*64+:64] = 64'hfedcba9876543210 ^ (64'h0123456789abcdef * (seed + i));
      return value;
    endfunction
    function void initialize(u64 addr, int seed);
      ram_ref_program(oracle, addr, pattern(seed));
      ram_ref_program(ddr, addr, pattern(seed));
    endfunction
    function void remember(command_t q, result_item expected);
      int key = (q.source << 12) | q.tag;
      ck(!commands.exists(key), "Duplicate live upstream tag");
      foreach (commands[k])
      ck(commands[k].done || (commands[k].address >> 6) != (q.address >> 6),
         "Conflicting request admitted before predecessor final response");
      expected.source = q.source;
      expected.tag = q.tag;
      expected.set_transaction_id(key);
      scoreboard.expected_export.write(expected);
      commands[key] = q;
      accepted++;
      if (commands.num() > peak) peak = commands.num();
    endfunction
    task service();
      forever begin
        @(control.sample);
        if (control.sample.reset) begin
          cancelled += commands.num();
          commands.delete();
          inflight.delete();
          pending0.delete();
          pending1.delete();
          busy[0] = 0;
          busy[1] = 0;
          scoreboard.expected.delete();
          scoreboard.actual.delete();
          ram_ref_reset(oracle);
          ram_ref_reset(ddr);
        end else begin
          cycle++;
          for (int port = 0; port < 2; port++)
          if (returned[port].sample.valid && returned[port].sample.ready) begin
`ifdef RAM_DDR
            int id = returned[port].sample.bits[`F(REPLY, ID)];
            ck(inflight.exists(id), "Observed bridge response without request");
            ck(returned[port].sample.bits[`F(REPLY, ERROR)] == inflight[id].bits[`F(REPLY, ERROR)],
               "Bridge error propagation mismatch");
            if (inflight[id].terminal) commands[inflight[id].key].done = 1;
            inflight.delete(id);
`else
            if (active[port].terminal) commands[active[port].key].done = 1;
            busy[port] = 0;
`endif
          end
          for (int i = 0; i < 2; i++) begin
            if (cpu[i].sample.valid && !cpu[i].sample.ready) stalls++;
            if (line[i].sample.valid && !line[i].sample.ready) stalls++;
            if (cpu[i].sample.valid && cpu[i].sample.ready) begin
              result_item expected = new;
              command_t q;
              u64 value;
              q.source = i;
              q.tag = cpu[i].sample.bits[`F(CPU, TAG)];
              q.address = cpu[i].sample.bits[`F(CPU, ADDR)];
              q.op = cpu[i].sample.bits[`F(CPU, ATOMIC)];
              q.size = cpu[i].sample.bits[`F(CPU, SIZE)];
              q.write = cpu[i].sample.bits[`F(CPU, WRITE)];
              q.starts = 0;
              q.fault = faults.exists(q.address & ~63) ? faults[q.address&~63] : 0;
              q.mask = ((64'h1 << (1 << q.size)) - 1) << (q.address & 63);
              expected.error = ram_ref_cpu(
                  oracle,
                  i,
                  q.address,
                  q.size,
                  q.write,
                  q.op,
                  cpu[i].sample.bits[
                  `F(CPU, DATA)
                  ],
                  q.fault,
                  value
              );
              expected.data = value;
              q.done=(q.op==11&&!expected.error&&value==1)||q.address<64'h80000000||q.address>=64'h81000000;
              if (q.op == 11) begin
                if (value == 1) sc_failure++;
                else if (!expected.error) sc_success++;
              end
              remember(q, expected);
            end
            if (line[i].sample.valid && line[i].sample.ready) begin
              result_item expected = new;
              command_t   q;
              q.source = i + 2;
              q.tag = line[i].sample.bits[`F(LINE, ID)];
              q.address = line[i].sample.bits[`F(LINE, ADDR)];
              q.op = 0;
              q.size = 6;
              q.write = line[i].sample.bits[`F(LINE, WRITE)];
              q.mask = line[i].sample.bits[`F(LINE, MASK)];
              q.starts = 0;
              q.done = 0;
              q.fault = faults.exists(q.address) ? faults[q.address] : 0;
              expected.error = (q.fault & (q.write ? 2 : 1)) != 0;
              if (q.write)
                ram_ref_line_write(oracle, q.address, line[i].sample.bits[`F(LINE, DATA)], q.mask,
                                   int'(expected.error));
              else if (!expected.error) ram_ref_read(oracle, q.address, expected.data);
              remember(q, expected);
            end
          end
          for (int port = 0; port < 2; port++)
          if (memory[port].sample.valid && memory[port].sample.ready) begin
            pending_t response;
            int found = -1;
            bit [511:0] data, expected;
            u64 addr = memory[port].sample.bits[`F(LINE, ADDR)];
            bit write = memory[port].sample.bits[`F(LINE, WRITE)];
            int id = memory[port].sample.bits[`F(LINE, ID)];
            foreach (commands[key])
            if (!commands[key].done && (commands[key].address & ~63) == addr) begin
              ck(found == -1, "Ambiguous lock owner");
              found = key;
            end
            ck(found >= 0, "Backend request has no accepted command");
            ck(id / 4 == port, "Backend ID crossed port namespace");
            response.bits = '0;
            response.bits[`F(REPLY, ID)] = id;
            response.bits[`F(REPLY, ERROR)] = (commands[found].fault & (write ? 2 : 1)) != 0;
            if (write) begin
              ck(commands[found].write || commands[found].op > 0,
                 "Unexpected write for read command");
              ck(memory[port].sample.bits[`F(LINE, MASK)] == commands[found].mask,
                 "Write mask mismatch");
              if (!response.bits[`F(REPLY, ERROR)]) begin
                ram_ref_read(oracle, addr, expected);
                data = memory[port].sample.bits[`F(LINE, DATA)];
                for (int byte_index = 0; byte_index < 64; byte_index++)
                if (commands[found].mask[byte_index])
                  ck(data[byte_index*8+:8] == expected[byte_index*8+:8],
                     "Byte/AMO oracle mismatch");
              end
`ifndef RAM_DDR
              ram_ref_line_write(ddr, addr, memory[port].sample.bits[`F(LINE, DATA)],
                                 memory[port].sample.bits[`F(LINE, MASK)], int'(response.bits[
                                 `F(REPLY, ERROR)]));
`endif
              backend_writes++;
            end else begin
              ram_ref_read(ddr, addr, data);
              response.bits[`F(REPLY, DATA)] = data;
              backend_reads++;
            end
            if (commands[found].source < 2 && commands[found].op >= 1 && commands[found].op <= 9)
              ck(write == (commands[found].starts == 1),
                 "RMW did not preserve read then write phases");
            else ck(commands[found].starts == 0, "Unexpected extra backend operation");
            commands[found].starts++;
            response.terminal = write || response.bits[
            `F(REPLY, ERROR)
            ] || commands[found].op == 0 || commands[found].op == 10;
            response.key = found;
            response.due = cycle + 8 + (id % 3);
`ifdef RAM_DDR
            ck(!inflight.exists(id), "Bridge caller ID reused while active");
            inflight[id] = response;
`else
            if (port == 0) pending0.push_back(response);
            else pending1.push_back(response);
`endif
          end
          for (int source = 0; source < 4; source++) begin
            bit valid, ready, error;
            int tag;
            bit [511:0] data;
            if (source < 2) begin
              valid = result[source].sample.valid;
              ready = result[source].sample.ready;
              tag   = result[source].sample.bits[`F(RESULT, TAG)];
              data  = result[source].sample.bits[`F(RESULT, DATA)];
              error = result[source].sample.bits[`F(RESULT, ERROR)];
            end else begin
              valid = reply[source-2].sample.valid;
              ready = reply[source-2].sample.ready;
              tag   = reply[source-2].sample.bits[`F(REPLY, ID)];
              data  = reply[source-2].sample.bits[`F(REPLY, DATA)];
              error = reply[source-2].sample.bits[`F(REPLY, ERROR)];
            end
            if (valid) begin
              int key = (source << 12) | tag;
              ck(commands.exists(key) && commands[key].done,
                 "Early response before final backend completion");
              if (ready) begin
                result_item actual = new;
                actual.set_transaction_id(key);
                actual.source = source;
                actual.tag = tag;
                actual.data = data;
                actual.error = error;
                scoreboard.actual_export.write(actual);
                checked++;
                if (error) errors++;
                commands.delete(key);
              end
            end
          end
        end
        @(negedge control.clock);
        for (int i = 0; i < 2; i++) begin
`ifdef RAM_DDR
          memory[i].ready   = 1;
          returned[i].ready = 1;
`else
          memory[i].ready = !control.reset && cycle % 5 != i;
`endif
          result[i].ready = !control.reset && !hold_cpu && cycle % 7 != i;
          reply[i].ready  = !control.reset && cycle % 7 != i;
`ifndef RAM_DDR
          if (!busy[i] && !hold_memory) begin
            if (i == 0)
              for (int q = 0; q < pending0.size(); q++)
              if(!busy[i]&&pending0[q].due<=cycle&&(held_line==0||(commands[pending0[q].key].address&~63)!=held_line))begin
                active[i] = pending0[q];
                pending0.delete(q);
                busy[i] = 1;
              end
            if (i == 1)
              for (int q = 0; q < pending1.size(); q++)
              if(!busy[i]&&pending1[q].due<=cycle&&(held_line==0||(commands[pending1[q].key].address&~63)!=held_line))begin
                active[i] = pending1[q];
                pending1.delete(q);
                busy[i] = 1;
              end
          end
          returned[i].valid = !control.reset && busy[i];
          returned[i].bits  = active[i].bits;
`endif
        end
      end
    endtask
    task cpu_send(int who, int tag, u64 addr, int op = 0, u64 value = 0, int size = 3,
                  bit write = 0);
      @(negedge control.clock);
      cpu[who].bits = '0;
      cpu[who].bits[`F(CPU, TAG)] = tag;
      cpu[who].bits[`F(CPU, ADDR)] = addr;
      cpu[who].bits[`F(CPU, ATOMIC)] = op;
      cpu[who].bits[`F(CPU, DATA)] = value;
      cpu[who].bits[`F(CPU, SIZE)] = size;
      cpu[who].bits[`F(CPU, WRITE)] = write;
      cpu[who].valid = 1;
      do @(cpu[who].sample); while (!cpu[who].sample.ready);
      @(negedge control.clock);
      cpu[who].valid = 0;
    endtask
    task line_send(int who, int tag, u64 addr, bit write = 0, bit [63:0] mask = '1);
      @(negedge control.clock);
      line[who].bits = '0;
      line[who].bits[`F(LINE, ID)] = tag;
      line[who].bits[`F(LINE, ADDR)] = addr;
      line[who].bits[`F(LINE, WRITE)] = write;
      line[who].bits[`F(LINE, DATA)] = pattern(tag);
      line[who].bits[`F(LINE, MASK)] = mask;
      line[who].valid = 1;
      do @(line[who].sample); while (!line[who].sample.ready);
      @(negedge control.clock);
      line[who].valid = 0;
    endtask
    task drain();
      wait (commands.num() == 0);
      repeat (3) @(control.sample);
      ck(
          control.sample.outstanding==0&&!pending0.size()&&!pending1.size()&&!busy[0]&&!busy[1]&&!inflight.num(),
          "Memory/contexts not drained");
    endtask
    task execute();
      control.reset = 1;
      for (int i = 0; i < 2; i++) begin
        cpu[i].valid = 0;
        cpu[i].bits = 0;
        line[i].valid = 0;
        line[i].bits = 0;
        result[i].ready = 0;
        reply[i].ready = 0;
        memory[i].ready = 0;
`ifndef RAM_DDR
        returned[i].valid = 0;
        returned[i].bits  = 0;
`endif
      end
      for (int i = 0; i < 128; i++) initialize(64'h80000000 + i * 64, i + 1);
`ifdef RAM_DDR
      axi_init();
      fork
        axi_service();
      join_none
`endif
      repeat (4) @(negedge control.clock);
      control.reset = 0;
      fork
        service();
      join_none
      hold_memory = 1;
      fork
        cpu_send(0, 0, 'h80000000);
        cpu_send(1, 0, 'h80000040);
        begin
          for (int i = 0; i < 3; i++) line_send(0, i, 'h80000100 + i * 64);
        end
        begin
          for (int i = 0; i < 3; i++) line_send(1, i, 'h80000400 + i * 64, 1);
        end
      join
      repeat (15) @(control.sample);
      ck(peak == 8 && checked == 0, "Failed eight nonconflicting outstanding slots");
      hold_memory = 0;
      drain();
      for (int size = 2; size <= 3; size++)
        for (int lane = 0; lane < (size == 2 ? 2 : 1); lane++)
          for (int op = 1; op <= 9; op++) begin
            cpu_send(op % 2, op, 'h80000800 + lane * 4, op, 64'h800000007fffffff, size);
            drain();
            line_send(0, 100 + op, 'h80000800);
            drain();
          end
      cpu_send(0, 1, 'h80000904, 10, 0, 2);
      drain();
      cpu_send(0, 2, 'h80000904, 11, 'h87654321, 2);
      drain();
      cpu_send(0, 3, 'h80000900, 10);
      drain();
      cpu_send(1, 3, 'h80000908, 0, 'h55, 3, 1);
      drain();
      cpu_send(0, 4, 'h80000900, 11, 'hdead);
      drain();
      cpu_send(0, 5, 'h80000900, 10);
      drain();
      line_send(1, 7, 'h80000900, 1, 64'hff00);
      drain();
      cpu_send(0, 6, 'h80000900, 11, 99);
      drain();
      cpu_send(0, 7, 'h80000900, 10);
      drain();
      line_send(0, 8, 'h80000900, 1, 0);
      drain();
      cpu_send(0, 8, 'h80000900, 11, 77);
      drain();
      // LR holds its line until the delayed read has established the reservation.
      hold_memory = 1;
      cpu_send(0, 9, 'h80000a00, 10);
      fork
        line_send(0, 9, 'h80000a00, 1);
        begin
          repeat (12) @(control.sample);
          ck(!line[0].sample.ready, "Writer bypassed in-flight LR");
          hold_memory = 0;
        end
      join
      drain();
      cpu_send(0, 10, 'h80000a00, 11, 88);
      drain();
      // A second CPU AMO cannot enter between the first AMO read and final write ACK.
      hold_memory = 1;
      cpu_send(0, 11, 'h80000b00, 2, 1);
      fork
        cpu_send(1, 11, 'h80000b00, 2, 1);
        begin
          repeat (12) @(control.sample);
          ck(!cpu[1].sample.ready, "AMO conflict bypassed line lock");
          hold_memory = 0;
        end
      join
      drain();
      for (int i = 0; i < 3; i++) begin
        faults['h80000c00+i*64] = i == 0 ? 1 : 2;
        cpu_send(i % 2, 20 + i, 'h80000c00 + i * 64, i == 2 ? 2 : 0, 9, 3, i == 1);
        drain();
      end
      cpu_send(0, 30, 'h81000000);
      drain();
      for (int size = 0; size < 4; size++) begin
        cpu_send(0, 32 + size, 'h80000d00 + (1 << size), 0, 64'hfedcba9876543210, size, 1);
        drain();
        cpu_send(1, 32 + size, 'h80000d00 + (1 << size), 0, 0, size);
        drain();
      end
      held_line = 'h80000e00;
      cpu_send(0, 42, held_line, 2, 1);
      cpu_send(1, 43, 'h80000e40);
      wait (!commands.exists((1 << 12) | 43));
      ck(commands.exists(42) && !commands[42].done,
         "Unrelated line did not progress around held AMO");
      parallel_progress++;
      held_line = 0;
      drain();
      hold_cpu = 1;
      cpu_send(0, 44, 'h80000f00);
      wait (result[0].valid);
      repeat (8) @(control.sample);
      hold_cpu = 0;
      drain();
      hold_memory = 1;
      cpu_send(0, 45, 'h80001000, 10);
      line_send(0, 45, 'h80001040);
      repeat (6) @(control.sample);
      @(negedge control.clock);
      control.reset = 1;
      repeat (3) @(negedge control.clock);
      control.reset = 0;
      hold_memory   = 0;
      cpu_send(0, 46, 'h80001000, 11, 123);
      drain();
`ifdef RAM_DDR
      held_write_line = 'h80001100;
      cpu_send(0, 47, held_write_line, 2, 1);
      wait (final_b_waits > 0);
      fork
        line_send(1, 47, 'h80001100, 1);
        begin
          repeat (15) @(control.sample);
          ck(!line[1].sample.ready && commands.exists(47) && !commands[47].done,
             "AMO released line before final AXI B");
          held_write_line = 0;
        end
      join
      drain();
      axi_drained();
`endif
      ck(
          accepted==checked+cancelled&&cancelled==2&&parallel_progress==1&&peak==8&&sc_success==2&&sc_failure==4&&errors==4&&stalls>0,
          "Missing atomic/error/concurrency evidence");
      `uvm_info(
          "UNCACHED_RAM_PASS",
          $sformatf(
              "accepted=%0d checked=%0d cancelled=%0d parallel=%0d peak=%0d LRSC success/fail=%0d/%0d errors=%0d backendR/W=%0d/%0d stalls=%0d; byte oracle and all contexts drained",
              accepted, checked, cancelled, parallel_progress, peak, sc_success, sc_failure,
              errors, backend_reads, backend_writes, stalls), UVM_LOW)
    endtask
`ifdef RAM_DDR
    `include "axi.svh"
`endif
    function void final_phase(uvm_phase phase);
      ram_ref_destroy(oracle);
      ram_ref_destroy(ddr);
      super.final_phase(phase);
    endfunction
  endclass
endpackage
