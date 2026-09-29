package bankset_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  import axis_vip_pkg::*;
  `include "uvm_macros.svh"
  typedef axis_item#(32) item;

  import "DPI-C" function chandle bank_ref_create(input int unsigned entries);
  import "DPI-C" function void bank_ref_destroy(input chandle model);
  import "DPI-C" function int unsigned bank_ref_access(
    input chandle model,
    input int unsigned addr,
    input byte unsigned write,
    input int unsigned data,
    input byte unsigned mask
  );

  class operation extends uvm_sequence_item;
    bit write;
    int unsigned word_addr;
    int unsigned data;
    byte unsigned mask;
    bit last;
    `uvm_object_utils(operation)
    function new(string name = "operation");
      super.new(name);
    endfunction
  endclass

  class ref_model extends reference_model #(operation, item);
    `uvm_component_utils(ref_model)
    chandle model;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      model = bank_ref_create(64);
    endfunction
    function void write(operation t);
      int unsigned result = bank_ref_access(model, t.word_addr, t.write, t.data, t.mask);
      if (!t.write) begin
        item expected = item::type_id::create("expected");
        expected.data = result;
        expected.keep = 15;
        expected.last = t.last;
        expected_ap.write(expected);
      end
    endfunction
    function void final_phase(uvm_phase phase);
      bank_ref_destroy(model);
      super.final_phase(phase);
    endfunction
  endclass

  class env extends checked_env #(operation, item, ref_model);
    `uvm_component_utils(env)
    virtual bankset_if control;
    virtual axis_if write_vif;
    axis_source_agent #(32) source;
    axis_sink #(32) sink;
    axis_monitor #(32) output_monitor;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(virtual bankset_if)::get(
              this, "", "control", control
          ) || !uvm_config_db#(virtual axis_if)::get(
              this, "", "write_vif", write_vif
          ))
        `uvm_fatal("VIF", "BankSet interfaces missing")
      source = axis_source_agent#(32)::type_id::create("source", this);
      sink = axis_sink#(32)::type_id::create("sink", this);
      output_monitor = axis_monitor#(32)::type_id::create("output_monitor", this);
    endfunction
    function void connect_phase(uvm_phase phase);
      super.connect_phase(phase);
      output_monitor.ap.connect(output_export);
    endfunction
    task run_phase(uvm_phase phase);
      int unsigned word_addr;
      operation op;
      forever begin
        @(posedge control.clock);
        if (!control.reset) begin
          if (control.valid && control.ready) begin
            word_addr = control.addr / 4;
            if (!control.write) begin
              for (int i = 0; i < control.beats; i++) begin
                op = operation::type_id::create("read");
                op.write = 0;
                op.word_addr = word_addr + i;
                op.last = i == control.beats - 1;
                input_export.write(op);
              end
            end
          end
          if (write_vif.tvalid && write_vif.tready) begin
            op = operation::type_id::create("write");
            op.write = 1;
            op.word_addr = word_addr;
            op.data = write_vif.tdata;
            op.mask = write_vif.tkeep;
            input_export.write(op);
            word_addr++;
          end
        end
      end
    endtask
  endclass

  class write_sequence extends uvm_sequence #(item);
    `uvm_object_utils(write_sequence)
    int unsigned beats;
    bit masked;
    function new(string name = "write_sequence");
      super.new(name);
    endfunction
    task body();
      item req;
      for (int i = 0; i < beats; i++) begin
        req = item::type_id::create("req");
        start_item(req);
        req.data = $urandom();
        req.keep = masked ? (i % 16) : 15;
        req.last = i == beats - 1;
        finish_item(req);
      end
    endtask
  endclass

  class protocol_test extends ip_test;
    `uvm_component_utils(protocol_test)
    env test_env;
    virtual bankset_if control;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      test_env = env::type_id::create("env", this);
      if (!uvm_config_db#(virtual bankset_if)::get(this, "", "control", control))
        `uvm_fatal("VIF", "BankSet control interface missing")
    endfunction
    task transfer(bit write, int unsigned word_addr, int unsigned beats, bit masked = 0);
      write_sequence seq;
      int unsigned   target = test_env.scoreboard.checked + (write ? 0 : beats);
      @(negedge control.clock);
      control.valid = 1;
      control.write = write;
      control.addr = word_addr * 4;
      control.beats = beats;
      control.done_ready = 0;
      do @(posedge control.clock); while (!control.ready);
      @(negedge control.clock);
      control.valid = 0;
      if (write) begin
        seq = write_sequence::type_id::create("seq");
        seq.beats = beats;
        seq.masked = masked;
        seq.start(test_env.source.seqr);
      end else begin
        test_env.scoreboard.wait_checked(target);
      end
      do @(negedge control.clock); while (!control.done_valid);
      repeat (3) begin
        if (control.done_valid !== 1'b1 || control.error !== 1'b0 || control.ready !== 1'b0)
          `uvm_fatal("DONE", "invalid BankSet completion under backpressure")
        @(negedge control.clock);
      end
      control.done_ready = 1;
      repeat (2) @(negedge control.clock);
      control.done_ready = 0;
      if (control.done_valid !== 1'b0 || control.ready !== 1'b1)
        `uvm_fatal("DONE", "BankSet failed to retire completion")
    endtask
    task execute();
      control.valid = 0;
      control.write = 0;
      control.addr = 0;
      control.beats = 0;
      control.done_ready = 0;
      wait (!control.reset);
      transfer(1, 0, 64);
      test_env.sink.stall_cycles = 0;
      transfer(0, 0, 63);
      test_env.sink.stall_cycles = 3;
      transfer(1, 63, 1);
      transfer(0, 63, 1);
      transfer(1, 3, 17, 1);
      control.reset = 1;
      repeat (2) @(negedge control.clock);
      control.reset = 0;
      transfer(0, 0, 64);
      `uvm_info("BANKSET", $sformatf("Checked %0d read beats", test_env.scoreboard.checked),
                UVM_LOW)
    endtask
  endclass
endpackage
