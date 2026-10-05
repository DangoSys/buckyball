class protocol_test extends ip_test;
  `uvm_component_utils(protocol_test)
  virtual rvv_if vif;
  kernel_env test_env;
  int unsigned completed = 0;
  bit protocol_fault = 0;
  bit reject_write = 0;
  bit [31:0] protocol_cause, protocol_tval, protocol_pc, protocol_instruction;
  int unsigned requests[4];
  function new(string name, uvm_component parent);
    super.new(name, parent);
    timeout = 20ms;
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual rvv_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "RVV interface missing")
    test_env = kernel_env::type_id::create("env", this);
  endfunction
  task memory_port(int p);
    bit read_pending = 0, write_pending = 0;
    int read_delay, write_delay;
    bit [31:0] seed = 32'h629731a5 ^ (p * 32'h1234567);
    int unsigned address;
    forever begin
      @(negedge vif.clock);
      seed = {seed[30:0], seed[31] ^ seed[21] ^ seed[1] ^ seed[0]};
      if (vif.reset) begin
        read_pending = 0;
        write_pending = 0;
        vif.read_ready[p] = 0;
        vif.write_ready[p] = 0;
        vif.read_response_valid[p] = 0;
        vif.write_response_valid[p] = 0;
      end else begin
        vif.read_ready[p]  = !read_pending && seed[0];
        vif.write_ready[p] = !write_pending && seed[1];
        if (read_pending && read_delay != 0) read_delay--;
        if (write_pending && write_delay != 0) write_delay--;
        vif.read_response_valid[p]  = read_pending && read_delay == 0;
        vif.write_response_valid[p] = write_pending && write_delay == 0;
      end
      @(posedge vif.clock);
      if (!vif.reset) begin
        if (vif.read_valid[p] && vif.read_ready[p]) begin
          requests[p]++;
          read_pending = 1;
          read_delay = 1 + seed[4:2];
          address = (int'(vif.read_bank[p]) << 16) | (int'(vif.read_row[p]) << 4);
          vif.read_data[p] = {rvv_memory_read(address + 8, 3), rvv_memory_read(address, 3)};
        end
        if (vif.write_valid[p] && vif.write_ready[p]) begin
          requests[p]++;
          write_pending = 1;
          write_delay = 1 + seed[7:5];
          address = (int'(vif.write_bank[p]) << 16) | (int'(vif.write_row[p]) << 4);
          if (!reject_write) begin
            rvv_memory_masked_write(address, vif.write_data[p][63:0], vif.write_mask[p][7:0]);
            rvv_memory_masked_write(address + 8, vif.write_data[p][127:64],
                                    vif.write_mask[p][15:8]);
          end
          vif.write_ok[p] = !reject_write;
        end
        if (vif.read_response_valid[p] && vif.read_response_ready[p]) read_pending = 0;
        if (vif.write_response_valid[p] && vif.write_response_ready[p]) write_pending = 0;
      end
    end
  endtask
  task monitor();
    forever begin
      @(posedge vif.clock);
      if (!vif.reset) begin
        if (vif.command_valid && vif.command_ready) begin
          launch_item item = launch_item::type_id::create("launch");
          item.execute = vif.command_funct7 == 15;
          item.protocol_fault = protocol_fault;
          item.protocol_cause = protocol_cause;
          item.protocol_tval = protocol_tval;
          item.protocol_pc = protocol_pc;
          item.protocol_instruction = protocol_instruction;
          item.rob = vif.command_rob;
          test_env.input_export.write(item);
        end
        if (vif.done_valid && vif.done_ready) begin
          completion_item item = completion_item::type_id::create("completion");
          item.fault = vif.done_fault;
          item.pc = vif.done_pc;
          item.instruction = vif.done_instruction;
          item.cause = vif.done_cause;
          item.tval = vif.done_tval;
          item.rob = vif.done_rob;
          item.write_bank = vif.done_write_bank;
          test_env.output_export.write(item);
        end
      end
    end
  endtask
  task command(int funct7, longint unsigned rs1, longint unsigned rs2);
    @(negedge vif.clock);
    vif.command_funct7 = funct7;
    vif.command_rs1 = rs1;
    vif.command_rs2 = rs2;
    vif.command_rob = completed % 16;
    vif.command_valid = 1;
    do @(posedge vif.clock); while (!vif.command_ready);
    @(negedge vif.clock);
    vif.command_valid = 0;
  endtask
  task retire();
    bit [218:0] held;
    while (!vif.done_valid) @(negedge vif.clock);
    held = {
      vif.done_rob,
      vif.done_write_bank,
      vif.done_fault,
      vif.done_pc,
      vif.done_instruction,
      vif.done_cycles,
      vif.done_cause,
      vif.done_tval,
      vif.done_fflags,
      vif.done_vxsat
    };
    repeat (5) begin
      @(negedge vif.clock);
      if(!vif.done_valid || held !== {vif.done_rob,vif.done_write_bank,vif.done_fault,vif.done_pc,
          vif.done_instruction,vif.done_cycles,vif.done_cause,vif.done_tval,vif.done_fflags,vif.done_vxsat})
        `uvm_fatal("COMPLETION", "completion changed under backpressure")
      if (vif.command_ready || !vif.busy)
        `uvm_fatal("COMPLETION", "completion allowed command before retirement")
    end
    vif.done_ready = 1;
    @(negedge vif.clock);
    vif.done_ready = 0;
    completed++;
    test_env.scoreboard.wait_checked(completed);
  endtask
  task upload();
    command(12, (longint'(rvv_case_meta(0)) << 32) | rvv_image_bytes(), 64'h10000000);
    for (int word_index = 0; word_index < rvv_image_bytes() / 4; word_index++) begin
      @(negedge vif.clock);
      vif.image_valid = 1;
      vif.image_data  = rvv_image_word(word_index);
      do @(posedge vif.clock); while (!vif.image_ready);
    end
    @(negedge vif.clock);
    vif.image_valid = 0;
    retire();
    if (!rvv_compare_image()) `uvm_fatal("IMAGE", "image load modified compute banks")
  endtask
  task execute();
    vif.reset = 1;
    vif.image_valid = 0;
    vif.command_valid = 0;
    vif.done_ready = 0;
    foreach (vif.read_ready[p]) begin
      vif.read_ready[p] = 0;
      vif.write_ready[p] = 0;
      vif.read_response_valid[p] = 0;
      vif.write_response_valid[p] = 0;
      vif.read_data[p] = 0;
      vif.write_ok[p] = 0;
    end
    rvv_model_init();
    fork
      memory_port(0);
      memory_port(1);
      memory_port(2);
      memory_port(3);
      monitor();
    join_none
    repeat (4) @(negedge vif.clock);
    vif.reset = 0;
    for (int case_index = 0; case_index < rvv_case_count(); case_index++) begin
      rvv_case_select(case_index);
      upload();
      command(15, 2048, rvv_case_meta(0));
      while (!vif.done_valid) @(negedge vif.clock);
      if (!rvv_compare_memory())
        `uvm_fatal("REFERENCE", $sformatf("case %0d bank contents differ", case_index))
      retire();
      `uvm_info("RVV", $sformatf("case %0d completed and compared", case_index), UVM_LOW)
    end
    foreach (requests[p])
      if (requests[p] == 0) `uvm_fatal("PORTS", $sformatf("bank port %0d was never exercised", p))
    // A malformed header reports a command fault before any code/data is written.
    protocol_fault = 1;
    protocol_cause = 2;
    protocol_tval  = 32'hdeadbeef;
    command(12, rvv_image_bytes(), 0);
    for (int index = 0; index < 6; index++) begin
      @(negedge vif.clock);
      vif.image_valid = 1;
      vif.image_data  = index == 0 ? 32'hdeadbeef : rvv_image_word(index);
      do @(posedge vif.clock); while (!vif.image_ready);
    end
    @(negedge vif.clock);
    vif.image_valid = 0;
    retire();
    protocol_tval = 25;
    command(12, 25, 0);
    retire();
    protocol_cause = 1;
    protocol_tval  = 0;
    command(15, 2048, 0);
    retire();
    protocol_cause = 2;
    protocol_tval  = 127;
    command(127, 0, 0);
    retire();
    // Writable globals and BSS are forbidden in a kernel image.
    for (int invalid_field = 3; invalid_field <= 5; invalid_field += 2) begin
      protocol_cause = 2;
      protocol_tval  = invalid_field == 3 ? 32'h50000 : 4;
      command(12, rvv_image_bytes(), 0);
      for (int index = 0; index < 6; index++) begin
        @(negedge vif.clock);
        vif.image_valid = 1;
        vif.image_data = index == invalid_field ?
            (invalid_field == 3 ? 32'h50000 : 4) : rvv_image_word(index);
        do @(posedge vif.clock); while (!vif.image_ready);
      end
      @(negedge vif.clock);
      vif.image_valid = 0;
      retire();
    end
    // A failed compute-bank write acknowledgement is an execution fault.
    protocol_fault = 0;
    rvv_case_select(14);
    upload();
    protocol_fault = 1;
    protocol_cause = 7;
    protocol_tval = 32'h10001;
    protocol_pc = 4;
    protocol_instruction = rvv_program_word(1);
    reject_write = 1;
    command(15, 2048, rvv_case_meta(0));
    retire();
    reject_write = 0;
    protocol_pc = 0;
    protocol_instruction = 0;
    protocol_fault = 0;
    // Reset cancels a command while it waits for a descriptor bank response.
    rvv_case_select(rvv_case_count() - 2);
    upload();
    command(15, 2048, rvv_case_meta(0));
    do @(posedge vif.clock); while (!(vif.read_valid[0] && vif.read_ready[0]));
    @(negedge vif.clock);
    vif.reset = 1;
    test_env.scoreboard.cancel_pending();
    repeat (3) @(negedge vif.clock);
    vif.reset = 0;
    repeat (5) begin
      @(negedge vif.clock);
      if (vif.done_valid || vif.busy) `uvm_fatal("RESET", "reset did not cancel pending kernel")
    end
    `uvm_info("RVV", $sformatf("checked %0d Blink commands", completed), UVM_LOW)
  endtask
endclass
