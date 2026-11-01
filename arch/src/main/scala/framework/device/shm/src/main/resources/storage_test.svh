class protocol_test extends base_test;
  `uvm_component_utils(protocol_test)
  storage_vif access[2];
  result_board #(transaction, 1) scoreboard;
  byte unsigned memory[8192];
  bit initialized[8192];
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
      if (access[0].reset) scoreboard.cancel_pending();
      else
        for (int i = 0; i < 2; i++) begin
          if (access[i].valid && access[i].ready) begin
            expected = transaction::type_id::create("expected");
            expected.set_transaction_id(i);
            for (int b = 0; b < 16; b++) begin
              if (access[i].write && access[i].mask[b]) begin
                memory[access[i].address+b] = access[i].data[b*8+:8];
                initialized[access[i].address+b] = 1;
              end else if (!access[i].write) begin
                if (!initialized[access[i].address+b])
                  `uvm_fatal("INIT", "Read before initialization")
                expected.data[b*8+:8] = memory[access[i].address+b];
              end
            end
            scoreboard.write_expected(expected);
          end
          if (access[i].response_valid && access[i].response_ready) begin
            actual = transaction::type_id::create("actual");
            actual.set_transaction_id(i);
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
    int checked;
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
  endtask
endclass
