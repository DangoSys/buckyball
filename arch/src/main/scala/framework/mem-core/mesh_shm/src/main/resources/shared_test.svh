class protocol_test extends ip_test;
  `uvm_component_utils(protocol_test)
  request_vif source[10];
  response_vif sink[10];
  keyed_scoreboard #(response) scoreboard;
  chandle model;
  virtual transfer_if transfer;
  function new(string name, uvm_component parent);
    super.new(name, parent);
    timeout = 500us;
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    scoreboard = keyed_scoreboard#(response)::type_id::create("scoreboard", this);
    model = mesh_banks_create();
    if (!uvm_config_db#(virtual transfer_if)::get(this, "", "transfer", transfer))
      `uvm_fatal("VIF", "transfer interface missing")
    for (int i = 0; i < 10; i++)
    if (!uvm_config_db#(request_vif)::get(
            this, "", $sformatf("source%0d", i), source[i]
        ) || !uvm_config_db#(response_vif)::get(
            this, "", $sformatf("sink%0d", i), sink[i]
        ))
      `uvm_fatal("VIF", "mesh interfaces missing")
  endfunction
  task monitor();
    response item;
    forever begin
      @(posedge source[0].clock);
      if (source[0].reset) begin
        scoreboard.expected.delete();
        scoreboard.actual.delete();
      end else
        for (int i = 0; i < 10; i++) begin
          if (source[i].tvalid && source[i].tready) begin
            item = response::type_id::create("expected");
            item.error = mesh_bank_access(
                model,
                source[i].tdest,
                source[i].tuser >> 1,
                source[i].tuser[0],
                source[i].tdata,
                source[i].tkeep,
                item.data
            );
            item.write = source[i].tuser[0];
            item.tag = source[i].tid;
            item.channel = i;
            item.set_transaction_id(i * 256 + item.tag);
            scoreboard.write_expected(item);
          end
          if (sink[i].tvalid && sink[i].tready) begin
            item = response::type_id::create("actual");
            item.data = sink[i].tdata;
            item.tag = sink[i].tid;
            item.error = sink[i].tuser[0];
            item.write = sink[i].tuser[1];
            item.channel = i;
            item.set_transaction_id(i * 256 + item.tag);
            scoreboard.write_actual(item);
          end
        end
    end
  endtask
  task access (int channel, int bank, int row, bit write, bit [15:0] mask, int tag, int stall);
    int target = scoreboard.checked + 1;
    @(negedge source[0].clock);
    source[channel].tvalid = 1;
    source[channel].tdest = bank;
    source[channel].tuser = (row << 1) | write;
    source[channel].tkeep = mask;
    source[channel].tlast = 1;
    source[channel].tid = tag;
    source[channel].tdata = {$urandom(), $urandom(), $urandom(), $urandom()};
    sink[channel].tready = 0;
    do @(posedge source[0].clock); while (!source[channel].tready);
    @(negedge source[0].clock);
    source[channel].tvalid = 0;
    while (!sink[channel].tvalid) @(negedge source[0].clock);
    repeat (stall) @(negedge source[0].clock);
    sink[channel].tready = 1;
    scoreboard.wait_checked(target);
  endtask
  task concurrent_reads();
    bit [9:0] pending = '1;
    int target = scoreboard.checked + 10;
    @(negedge source[0].clock);
    for (int i = 0; i < 10; i++) begin
      source[i].tvalid = 1;
      source[i].tdest = i;
      source[i].tuser = (i + 3) << 1;
      source[i].tkeep = 0;
      source[i].tlast = 1;
      source[i].tid = 200 + i;
      sink[i].tready = 0;
    end
    fork
      begin
        repeat (5) @(negedge source[0].clock);
        for (int i = 0; i < 10; i++) begin
          sink[i].tready = 1;
          @(negedge source[0].clock);
        end
      end
    join_none
    while (pending) begin
      @(posedge source[0].clock);
      for (int i = 0; i < 10; i++) if (pending[i] && source[i].tready) pending[i] = 0;
      @(negedge source[0].clock);
      for (int i = 0; i < 10; i++) source[i].tvalid = pending[i];
    end
    scoreboard.wait_checked(target);
  endtask
  task outstanding_reads();
    int target = scoreboard.checked + 4;
    @(negedge source[0].clock);
    sink[0].tready = 0;
    for (int bank = 0; bank < 4; bank++) begin
      source[0].tvalid = 1;
      source[0].tdest = bank;
      source[0].tuser = (bank + 3) << 1;
      source[0].tkeep = 0;
      source[0].tlast = 1;
      source[0].tid = 220 + bank;
      do @(posedge source[0].clock); while (!source[0].tready);
      @(negedge source[0].clock);
      source[0].tvalid = 0;
    end
    // All four requests must be accepted before any response is consumed.
    repeat (8) @(negedge source[0].clock);
    sink[0].tready = 1;
    scoreboard.wait_checked(target);
  endtask
  task reset_read();
    @(negedge source[0].clock);
    source[0].tvalid = 1;
    source[0].tdest = 0;
    source[0].tuser = 6;
    source[0].tid = 254;
    sink[0].tready = 0;
    do @(posedge source[0].clock); while (!source[0].tready);
    @(negedge source[0].clock);
    source[0].tvalid = 0;
    while (!sink[0].tvalid) @(negedge source[0].clock);
    for (int i = 0; i < 10; i++) begin
      source[i].reset = 1;
      sink[i].reset   = 1;
    end
    repeat (3) @(negedge source[0].clock);
    for (int i = 0; i < 10; i++) begin
      source[i].reset = 0;
      sink[i].reset   = 0;
      sink[i].tready  = 1;
    end
    repeat (4) begin
      @(negedge source[0].clock);
      for (int i = 0; i < 10; i++)
      if (sink[i].tvalid) `uvm_fatal("RESET", "cancelled response survived reset")
    end
    access (0, 0, 3, 0, 0, 253, 1);
  endtask
  task move_rows(int source_core, int source_bank, int source_row, int target_core, int target_bank,
                 int target_row, int rows, bit invalid);
    bit [7:0] token = $urandom();
    @(negedge transfer.clock);
    transfer.valid = 1;
    transfer.source_core = source_core;
    transfer.source_bank = source_bank;
    transfer.source_row = source_row;
    transfer.target_core = target_core;
    transfer.target_bank = target_bank;
    transfer.target_row = target_row;
    transfer.rows = rows;
    transfer.tag = token;
    transfer.complete_ready = 0;
    if (!invalid)
      for (int i = 0; i < rows; i++) begin
        transfer.memory[source_core][source_bank][source_row+i] = {
          $urandom(), $urandom(), $urandom(), $urandom()
        };
        transfer.initialized[source_core][source_bank][source_row+i] = 1;
      end
    do @(posedge transfer.clock); while (!transfer.ready);
    @(negedge transfer.clock);
    transfer.valid = 0;
    while (!transfer.complete_valid) @(negedge transfer.clock);
    repeat (8) begin
      if (!transfer.complete_valid || transfer.complete_tag != token || transfer.error != invalid)
        `uvm_fatal("MVOVER", "completion changed or status/tag incorrect")
      @(negedge transfer.clock);
    end
    if (!invalid)
      for (int i = 0; i < rows; i++)
        if(!transfer.initialized[target_core][target_bank][target_row+i] ||
         transfer.memory[target_core][target_bank][target_row+i]!==transfer.memory[source_core][source_bank][source_row+i])
          `uvm_fatal("MVOVER", "target rows differ from source")
    transfer.complete_ready = 1;
    @(posedge transfer.clock);
    @(negedge transfer.clock);
  endtask
  task execute();
    transfer.valid = 0;
    transfer.complete_ready = 0;
    fork
      monitor();
    join_none
    for (int i = 0; i < 10; i++) begin
      source[i].reset = 1;
      source[i].tvalid = 0;
      sink[i].reset = 1;
      sink[i].tready = 0;
    end
    repeat (4) @(negedge source[0].clock);
    for (int i = 0; i < 10; i++) begin
      source[i].reset = 0;
      sink[i].reset   = 0;
    end
    for (int bank = 0; bank < 12; bank++) begin
      access (bank % 10, bank, bank + 3, 1, '1, bank, 3);
      access ((bank + 1) % 10, bank, bank + 3, 0, 0, bank, 5);
      for (int byte_index = 0; byte_index < 16; byte_index++) begin
        access (bank % 10, bank, bank + 3, 1, 16'b1 << byte_index, byte_index, 0);
        access ((bank + 2) % 10, bank, bank + 3, 0, 0, byte_index, 0);
      end
    end
    access (0, 12, 0, 0, 0, 250, 3);
    access (0, 0, 1023, 1, '1, 251, 2);
    access (0, 0, 1023, 0, 0, 252, 2);
    concurrent_reads();
    outstanding_reads();
    reset_read();
    move_rows(1, 1, 3, 2, 8, 9, 8, 0);
    move_rows(4, 0, 2040, 1, 15, 2040, 8, 0);
    move_rows(3, 1, 0, 3, 8, 4, 4, 0);
    move_rows(250, 0, 0, 2, 0, 0, 1, 1);
    move_rows(1, 0, 65530, 2, 0, 0, 9, 1);
    move_rows(1, 0, 0, 2, 0, 65530, 9, 1);
    move_rows(1, 0, 2048, 2, 0, 0, 1, 1);
    move_rows(1, 0, 0, 2, 0, 0, 0, 1);
    move_rows(1, 16, 0, 2, 0, 0, 1, 1);
    move_rows(1, 1, 3, 2, 16, 0, 1, 1);
    move_rows(1, 1023, 0, 2, 0, 0, 1, 1);
    for (int core = 0; core < 5; core++)
      move_rows(core, core, 16, (core + 1) % 5, 15 - core, 32, 8, 0);
    move_rows(1, 1, 0, 1, 1, 1, 8, 1);
    `uvm_info("MVOVER", "checked core1.bank1 to core2.bank8 and range/error cases", UVM_LOW)
    `uvm_info("MESH", $sformatf("checked %0d memory responses", scoreboard.checked), UVM_LOW)
  endtask
  function void final_phase(uvm_phase phase);
    mesh_banks_destroy(model);
    super.final_phase(phase);
  endfunction
endclass
