class protocol_test extends base_test;
  `uvm_component_utils(protocol_test)
  storage_vif access[2];
  result_board #(transaction, 1) scoreboard;
  byte unsigned memory[32768];
  bit initialized[32768];
  int issued[2], responded[2];
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    scoreboard = result_board#(transaction, 1)::type_id::create("scoreboard", this);
    for (int i = 0; i < 2; i++)
    if (!uvm_config_db#(storage_vif)::get(this, "", $sformatf("access%0d", i), access[i]))
      `uvm_fatal("VIF", "Missing device storage interface")
  endfunction
  task monitor();
    transaction expected, actual;
    forever begin
      @(posedge access[0].clock);
      if (access[0].reset) begin
        scoreboard.cancel_pending();
        foreach (issued[i]) begin
          issued[i] = 0;
          responded[i] = 0;
        end
      end else
        for (int i = 0; i < 2; i++) begin
          if (access[i].valid && access[i].ready) begin
            expected = transaction::type_id::create("expected");
            expected.set_transaction_id(i * 100000 + issued[i]++);
            for (int b = 0; b < 16; b++) begin
              if (access[i].write && access[i].mask[b]) begin
                memory[access[i].address+b] = access[i].data[b*8+:8];
                initialized[access[i].address+b] = 1;
              end else if (!access[i].write) begin
                if (!initialized[access[i].address+b])
                  `uvm_fatal("INIT", $sformatf(
                             "Read before initialization: client=%0d address=%0h byte=%0d",
                             i,
                             access[i].address,
                             b
                             ))
                expected.data[b*8+:8] = memory[access[i].address+b];
              end
            end
            scoreboard.write_expected(expected);
          end
          if (access[i].response_valid && access[i].response_ready) begin
            actual = transaction::type_id::create("actual");
            actual.set_transaction_id(i * 100000 + responded[i]++);
            actual.data  = access[i].result;
            actual.error = access[i].error;
            scoreboard.write_actual(actual);
          end
        end
    end
  endtask
  task send(int client, int address, bit write, bit [127:0] data, bit [15:0] mask);
    @(negedge access[0].clock);
    access[client].valid = 1;
    access[client].address = address;
    access[client].write = write;
    access[client].data = data;
    access[client].mask = mask;
    do @(posedge access[0].clock); while (!access[client].ready);
    @(negedge access[0].clock);
    access[client].valid = 0;
  endtask
  task execute();
    int checked, span;
    for (int i = 0; i < 2; i++) begin
      access[i].reset = 1;
      access[i].valid = 0;
      access[i].response_ready = 1;
    end
    fork
      monitor();
    join_none
    repeat (4) @(negedge access[0].clock);
    for (int i = 0; i < 2; i++) access[i].reset = 0;
    for (int row = 0; row < 8; row++) begin
      send(row % 2, row * 16, 1, {$urandom(), $urandom(), $urandom(), $urandom()}, '1);
      scoreboard.wait_checked(row + 1);
    end
    for (int row = 0; row < 8; row++) begin
      send(row % 2, row * 16, 1, '1, 16'h5555);
      scoreboard.wait_checked(9 + row * 2);
      send(1 - row % 2, row * 16, 0, 0, 0);
      scoreboard.wait_checked(10 + row * 2);
    end
    checked = scoreboard.checked;
    @(negedge access[0].clock);
    access[0].response_ready = 0;
    send(0, 0, 1, 128'h12345678, '1);
    send(1, 16, 0, 0, 0);
    scoreboard.wait_checked(checked + 1);
    repeat (4) begin
      @(negedge access[0].clock);
      if (!access[0].response_valid || access[0].result != 0 || access[0].error)
        `uvm_fatal("STALL", "Held response changed")
      evidence::checks++;
    end
    access[0].response_ready = 1;
    scoreboard.wait_checked(checked + 2);
    send(1, 0, 0, 0, 0);
    scoreboard.wait_checked(checked + 3);
    checked = scoreboard.checked;
    access[0].response_ready = 0;
    for (int i = 0; i < 4; i++) send(0, 256 + 16 * i, 1, 128'(i + 1), '1);
    repeat (4) begin
      @(negedge access[0].clock);
      if (access[0].ready || !access[0].response_valid)
        `uvm_fatal("CREDIT", "Four held responses must exhaust this client's credits")
      evidence::checks++;
    end
    access[0].response_ready = 1;
    scoreboard.wait_checked(checked + 4);
    checked = scoreboard.checked;
    span = 0;
    for (int i = 0; i < 32; i++) begin
      @(negedge access[0].clock);
      access[0].valid = 1;
      access[0].address = 512 + 16 * (i / 2) + (i % 2) * 16384;
      access[0].write = 1;
      access[0].data = 128'(i + 123);
      access[0].mask = '1;
      do begin
        @(posedge access[0].clock);
        span++;
      end while (!access[0].ready);
    end
    @(negedge access[0].clock);
    access[0].valid = 0;
    scoreboard.wait_checked(checked + 32);
    `uvm_info("THROUGHPUT", $sformatf("32 ordered SRAM writes accepted in %0d cycles", span),
              UVM_LOW)
    checked = scoreboard.checked;
    for (int i = 0; i < 32; i++) send(1, 512 + 16 * (i / 2) + (i % 2) * 16384, 0, 0, 0);
    scoreboard.wait_checked(checked + 32);
    checked = scoreboard.checked;
    access[0].response_ready = 0;
    for (int i = 0; i < 4; i++) send(0, 256 + 16 * i, 0, 0, 0);
    repeat (4) begin
      @(negedge access[0].clock);
      if (!access[0].response_valid || access[0].result != 1 || access[0].error)
        `uvm_fatal("ORDER", "Pipelined read response changed under backpressure")
      evidence::checks++;
    end
    access[0].response_ready = 1;
    scoreboard.wait_checked(checked + 4);
  endtask
endclass
