class protocol_test extends ip_test;
  `uvm_component_utils(protocol_test)
  router_vif source[5], sink[5];
  keyed_scoreboard #(packet) scoreboard;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    scoreboard = keyed_scoreboard#(packet)::type_id::create("scoreboard", this);
    for (int i = 0; i < 5; i++) begin
      if (!uvm_config_db#(router_vif)::get(
              this, "", $sformatf("source%0d", i), source[i]
          ) || !uvm_config_db#(router_vif)::get(
              this, "", $sformatf("sink%0d", i), sink[i]
          ))
        `uvm_fatal("VIF", "router interfaces missing")
    end
  endfunction
  function packet sample (router_vif vif, int port);
    packet p = packet::type_id::create("packet");
    p.data = vif.tdata;
    p.keep = vif.tkeep;
    p.last = vif.tlast;
    p.id = vif.tid;
    p.destination = vif.tdest;
    p.events = vif.tuser;
    p.port = port;
    p.set_transaction_id(p.id);
    return p;
  endfunction
  task monitor();
    packet p;
    forever begin
      @(posedge source[0].clock);
      if (source[0].reset) begin
        scoreboard.expected.delete();
        scoreboard.actual.delete();
      end else
        for (int i = 0; i < 5; i++) begin
          if (source[i].tvalid && source[i].tready) begin
            p = sample (source[i], 0);
            p.port = mesh_route(p.destination, 1, 1);
            scoreboard.write_expected(p);
          end
          if (sink[i].tvalid && sink[i].tready) scoreboard.write_actual(sample (sink[i], i));
        end
    end
  endtask
  task batch(int index, int destination, bit [4:0] active = 31);
    int target = scoreboard.checked + $countones(active);
    bit [4:0] pending = active;
    @(negedge source[0].clock);
    for (int i = 0; i < 5; i++) begin
      source[i].tvalid = active[i];
      source[i].tdata = {$urandom(), $urandom(), $urandom(), $urandom()};
      source[i].tkeep = index % 2 ? '1 : (16'b1 << (index % 16));
      source[i].tlast = 1;
      source[i].tid = index * 5 + i;
      source[i].tdest = destination;
      source[i].tuser = {$urandom(), $urandom()};
      sink[i].tready = 0;
    end
    fork
      begin
        repeat (4) @(negedge source[0].clock);
        for (int i = 0; i < 5; i++) sink[i].tready = 1;
      end
    join_none
    while (pending) begin
      @(posedge source[0].clock);
      for (int i = 0; i < 5; i++) if (pending[i] && source[i].tready) pending[i] = 0;
      @(negedge source[0].clock);
      for (int i = 0; i < 5; i++) source[i].tvalid = pending[i];
    end
    scoreboard.wait_checked(target);
  endtask
  task reset_queued();
    @(negedge source[0].clock);
    for (int i = 0; i < 5; i++) sink[i].tready = 0;
    source[0].tvalid = 1;
    source[0].tid = 250;
    source[0].tdest = 0;
    do @(posedge source[0].clock); while (!source[0].tready);
    @(negedge source[0].clock);
    source[0].tvalid = 0;
    repeat (2) @(negedge source[0].clock);
    if (!sink[2].tvalid) `uvm_fatal("RESET", "expected queued packet before reset")
    for (int i = 0; i < 5; i++) begin
      source[i].reset = 1;
      sink[i].reset   = 1;
    end
    repeat (2) @(negedge source[0].clock);
    for (int i = 0; i < 5; i++) begin
      source[i].reset = 0;
      sink[i].reset   = 0;
      sink[i].tready  = 1;
    end
    repeat (3) begin
      @(negedge source[0].clock);
      for (int i = 0; i < 5; i++)
      if (sink[i].tvalid) `uvm_fatal("RESET", "reset did not discard queued packet")
    end
  endtask
  task execute();
    fork
      monitor();
    join_none
    for (int i = 0; i < 5; i++) begin
      source[i].reset = 1;
      sink[i].reset = 1;
      source[i].tvalid = 0;
      sink[i].tready = 0;
    end
    repeat (4) @(negedge source[0].clock);
    for (int i = 0; i < 5; i++) begin
      source[i].reset = 0;
      sink[i].reset   = 0;
    end
    for (int iteration = 0; iteration < 4; iteration++)
      for (int row = 0; row < 3; row++)
        for (int col = 0; col < 4; col++) batch(iteration * 12 + row * 4 + col, row * 4 + col);
    for (int mask = 1; mask < 32; mask++)
      for (int row = 0; row < 3; row++)
        for (int col = 0; col < 4; col++) batch(mask, row * 4 + col, mask);
    reset_queued();
    batch(49, 5);
    `uvm_info("MESH", $sformatf("checked %0d routed packets", scoreboard.checked), UVM_LOW)
  endtask
endclass
