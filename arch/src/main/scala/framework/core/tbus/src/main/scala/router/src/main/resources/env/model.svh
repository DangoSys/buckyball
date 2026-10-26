class model extends uvm_subscriber #(cycle_item);
  `uvm_component_utils(model)
  localparam int BANKS = `TR_BANKS;
  fixture image;
  uvm_analysis_port #(result_item) expected, actual;
  uvm_analysis_port #(uvm_sequence_item) cancelled;
  const int clients[`TR_CLIENTS][17] = `TR_CLIENT_OFFSETS;
  const int mesh[`TR_CLIENTS+`TR_EXTERNAL][15] = `TR_MESH_OFFSETS;
  const int mappings[`TR_THREADS][10] = `TR_MAPPING_OFFSETS;
  const int owners[`TR_THREADS] = `TR_OWNERS;
  const int copies[`TR_LOCAL_FIELDS][4] = `TR_LOCAL_COPIES;
  bit live[BANKS];
  int owner[BANKS], architecture[BANKS], alias_id[BANKS], group_id[BANKS], leases[BANKS];
  int layouts[`TR_THREADS][`TR_ALIASES], committed[`TR_THREADS][`TR_ALIASES];
  logic [127:0] memory[BANKS][`TR_ROWS];
  bit [15:0] initialized[BANKS][`TR_ROWS];
  bit pending[`TR_CLIENTS+`TR_EXTERNAL], writing[`TR_CLIENTS+`TR_EXTERNAL];
  int
      selected[`TR_CLIENTS+`TR_EXTERNAL],
      rows[`TR_CLIENTS+`TR_EXTERNAL],
      tags[`TR_CLIENTS+`TR_EXTERNAL];
  bit accepted_now[`TR_CLIENTS+`TR_EXTERNAL];
  logic [127:0] data_now[`TR_CLIENTS+`TR_EXTERNAL];
  bit [15:0] masks_now[`TR_CLIENTS+`TR_EXTERNAL];
  bit lease_pending, move_pending;
  covergroup behavior with function sample (
      int port, bit writing, int owner, int event_kind, bit error, bit blocked
  );
    option.per_instance = 1;
    ports: coverpoint port {bins all[] = {[0 : `TR_CLIENTS + `TR_EXTERNAL - 1]};}
    direction: coverpoint writing;
    tenants: coverpoint owner {bins all[] = {[0 : `TR_THREADS - 1]};}
    events: coverpoint event_kind {
      bins access = {0};
      bins configuration = {1};
      bins renamed = {2};
      bins retirement = {3};
      bins export_lease = {4};
      bins release_lease = {5};
      bins move = {6};
      bins transfer = {7};
    }
    errors: coverpoint error;
    backpressure: coverpoint blocked;
  endgroup
  covergroup epochs with function sample (bit pending);
    option.per_instance = 1;
    cancellation: coverpoint pending;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    expected = new("expected", this);
    actual = new("actual", this);
    cancelled = new("cancelled", this);
    behavior = new();
    epochs = new();
  endfunction
  function void verify(bit ok, string reason);
    evidence::checks++;
    if (!ok) `uvm_fatal("TBUS_ROUTER", reason)
  endfunction
  function int tenant(longint unsigned id);
    foreach (owners[t]) if (owners[t] == id) return t;
    `uvm_fatal("OWNER", "Unknown configured owner")
    return -1;
  endfunction
  function int route(int thread, int tag, int group_index);
    int found = -1;
    foreach (live[b])
    if (live[b] && owner[b] == thread && alias_id[b] == tag && group_id[b] == group_index) begin
      verify(found == -1, "Ambiguous shared binding");
      found = b;
    end
    verify(found >= 0, "Missing shared binding");
    return found;
  endfunction
  function void request(int port, int bank_id, int row, bit wr, logic [127:0] data,
                        bit [15:0] mask);
    result_item result = result_item::type_id::create("expected");
    verify(!pending[port] && row < `TR_ROWS, "Overlapping or out-of-range shared access");
    result.port  = port;
    result.kind  = port < `TR_CLIENTS ? 0 : 1;
    result.error = image.error[port];
    result.set_transaction_id(port);
    selected[port] = bank_id;
    rows[port] = row;
    data_now[port] = data;
    masks_now[port] = mask;
    writing[port] = wr;
    accepted_now[port] = 1;
    pending[port] = 1;
    if (wr) begin
      if (!image.error[port])
        for (int b = 0; b < 16; b++)
        if (mask[b]) begin
          memory[bank_id][row][b*8+:8] = data[b*8+:8];
          initialized[bank_id][row][b] = 1;
        end
      result.data = 0;
    end else begin
      verify(initialized[bank_id][row] == '1, "Read of uninitialized shared memory");
      result.data = memory[bank_id][row];
    end
    expected.write(result);
    behavior.sample(port, wr, owner[bank_id], 0, image.error[port], 0);
  endfunction
  function void write(cycle_item t);
    if (t.reset) begin
      bit unfinished = lease_pending || move_pending;
      foreach (pending[c]) unfinished |= pending[c];
      epochs.sample(unfinished);
      cancelled.write(null);
      foreach (live[b]) begin
        live[b]   = 0;
        leases[b] = 0;
      end
      foreach (layouts[o, a]) begin
        layouts[o][a]   = 0;
        committed[o][a] = a;
      end
      foreach (pending[c]) begin
        pending[c] = 0;
        tags[c] = 0;
      end
      lease_pending = 0;
      move_pending  = 0;
      return;
    end
    foreach (accepted_now[c]) accepted_now[c] = 0;
    for (int c = 0; c < `TR_CLIENTS; c++) begin
      bit rd = t.inputs[clients[c][0]] && t.outputs[clients[c][1]];
      bit wr = t.inputs[clients[c][6]] && t.outputs[clients[c][7]];
      verify(!(rd && wr), "Client accepted both read and write");
      if (rd || wr) begin
        int thread = tenant(t.inputs[clients[c][16]+:64]);
        int alias_value = int'(t.inputs[clients[c][14]+:`TR_IN_MEM_REQ_0_BANK_ID_WIDTH]);
        int group_index = int'(t.inputs[clients[c][15]+:`TR_IN_MEM_REQ_0_GROUP_ID_WIDTH]);
        request(c, route(thread, alias_value, group_index),
                int'(t.inputs[clients[c][wr?8 : 2]+:16]), wr, t.inputs[clients[c][9]+:128],
                t.inputs[clients[c][10]+:16]);
      end
      if(t.outputs[clients[c][3]]&&t.inputs[clients[c][4]]||t.outputs[clients[c][11]]&&t.inputs[clients[c][12]])begin
        result_item result = result_item::type_id::create("actual");
        verify(pending[c], "Client response has no owner");
        result.port  = c;
        result.kind  = 0;
        result.data  = writing[c] ? 128'd0 : t.outputs[clients[c][5]+:128];
        result.error = writing[c] ? !t.outputs[clients[c][13]] : 0;
        result.set_transaction_id(c);
        actual.write(result);
        pending[c] = 0;
      end
      if(t.outputs[clients[c][3]]&&!t.inputs[clients[c][4]]||t.outputs[clients[c][11]]&&!t.inputs[clients[c][12]])
        behavior.sample(c, writing[c], 0, 0, 0, 1);
    end
    if (`RI(t, PHYSICAL_0_REQUEST_VALID) && `RO(t, PHYSICAL_0_REQUEST_READY)) begin
      longint unsigned address = `RI(t, PHYSICAL_0_REQUEST_BITS_ADDRESS);
      int bank_id = int'(address / (`TR_ROWS * 16));
      int row = int'(address % (16 * `TR_ROWS)) / 16;
      verify(address % 16 == 0 && bank_id < BANKS && live[bank_id],
             "Invalid physical shared access");
      request(`TR_CLIENTS, bank_id, row, `RI(t, PHYSICAL_0_REQUEST_BITS_WRITE),
              `RI(t, PHYSICAL_0_REQUEST_BITS_DATA), `RI(t, PHYSICAL_0_REQUEST_BITS_MASK));
    end
    if (`RO(t, PHYSICAL_0_RESPONSE_VALID) && `RI(t, PHYSICAL_0_RESPONSE_READY)) begin
      result_item result = result_item::type_id::create("physical_actual");
      verify(pending[`TR_CLIENTS], "Physical response has no owner");
      result.port  = `TR_CLIENTS;
      result.kind  = 1;
      result.data  = `RO(t, PHYSICAL_0_RESPONSE_BITS_DATA);
      result.error = `RO(t, PHYSICAL_0_RESPONSE_BITS_ERROR);
      result.set_transaction_id(`TR_CLIENTS);
      actual.write(result);
      pending[`TR_CLIENTS] = 0;
    end
    for (int c = 0; c < `TR_CLIENTS + `TR_EXTERNAL; c++) begin
      if (t.outputs[mesh[c][0]] && t.inputs[mesh[c][1]]) begin
        verify(accepted_now[c], "Mesh request has no accepted client");
        verify(t.outputs[mesh[c][2]+:$clog2(BANKS
               )] == selected[c] && t.outputs[mesh[c][3]+:17] == ((rows[c] << 1) | writing[c]),
               "Shared bank, row or direction routing mismatch");
        verify(t.outputs[mesh[c][7]], "Shared row request did not terminate its packet");
        if (c < `TR_CLIENTS) begin
          verify(t.outputs[mesh[c][6]+:`TR_TAG_BITS] == tags[c],
                 "Shared channel tag did not advance");
          tags[c] = (tags[c] + 1) % (1 << `TR_TAG_BITS);
        end else verify(t.outputs[mesh[c][6]+:`TR_TAG_BITS] == 0, "Physical port tag must be zero");
        if (writing[c])
          verify(
              t.outputs[mesh[c][4]+:128]===data_now[c]&&t.outputs[mesh[c][5]+:16]==masks_now[c],
              "Shared write data or mask changed");
        accepted_now[c] = 0;
      end
      verify(!accepted_now[c], "Client accepted without forwarding its Mesh request");
    end
    for (int thread = 0; thread < `TR_THREADS; thread++) begin
      if (t.inputs[mappings[thread][0]]) begin
        int previous = int'(t.inputs[mappings[thread][4]+:`TR_IN_MAPPING_0_BITS_PREVIOUS_WIDTH]);
        bit held;
        held = 0;
        foreach (live[b])
        if (live[b] && owner[b] == thread && alias_id[b] == previous && leases[b] != 0) held = 1;
        if (held && (t.inputs[mappings[thread][7]] || t.inputs[mappings[thread][8]])) begin
          verify(!t.outputs[mappings[thread][1]], "Mapping reclaimed a live export");
          behavior.sample(0, 0, thread, 3, 0, 1);
        end
      end
      if (t.inputs[mappings[thread][0]] && t.outputs[mappings[thread][1]]) begin
        int bank_id = int'(t.inputs[mappings[thread][2]+:`TR_IN_MAPPING_0_BITS_BANK_WIDTH]);
        int tag = int'(t.inputs[mappings[thread][3]+:`TR_IN_MAPPING_0_BITS_VBANK_WIDTH]);
        int previous = int'(t.inputs[mappings[thread][4]+:`TR_IN_MAPPING_0_BITS_PREVIOUS_WIDTH]);
        verify(t.inputs[mappings[thread][5]+:64] == owners[thread],
               "Mapping port changed its owner");
        if (t.inputs[mappings[thread][6]]) begin
          bit free_bank[BANKS];
          foreach (live[b])
          free_bank[b]=!live[b]||(t.inputs[mappings[thread][7]]&&owner[b]==thread&&alias_id[b]==previous);
          for (int group_index = 0; group_index < layouts[thread][bank_id]; group_index++) begin
            int chosen = -1;
            foreach (free_bank[b]) if (free_bank[b] && chosen < 0) chosen = b;
            verify(chosen >= 0, "Alias allocation exceeded capacity");
            free_bank[chosen] = 0;
            verify(leases[chosen] == 0, "Replaced a leased shared bank");
            live[chosen] = 1;
            owner[chosen] = thread;
            architecture[chosen] = bank_id;
            alias_id[chosen] = tag;
            group_id[chosen] = group_index;
          end
          behavior.sample(0, 0, thread, 2, 0, 0);
        end
        if (t.inputs[mappings[thread][8]]) begin
          committed[thread][bank_id] = tag;
          foreach (live[b])
          if (live[b] && owner[b] == thread && alias_id[b] == previous) live[b] = 0;
          behavior.sample(0, 0, thread, 3, 0, 0);
        end
      end
    end
    if (`RI(t, CONFIG_VALID) && `RO(t, CONFIG_READY)) begin
      int thread = tenant(`RI(t, CONFIG_BITS_THREAD_ID));
      int bank_id = `RI(t, CONFIG_BITS_VBANK_ID);
      int group_index = `RI(t, CONFIG_BITS_GROUP_ID);
      if (`RI(t, CONFIG_BITS_TRANSFER)) begin
        int source = `RI(t, CONFIG_BITS_SOURCE_BANK_ID);
        int prior = layouts[thread][bank_id];
        foreach (live[b])
        if (live[b] && owner[b] == thread && architecture[b] == source) begin
          verify(leases[b] == 0, "Transferred a leased bank");
          alias_id[b] = bank_id;
          architecture[b] = bank_id;
          group_id[b] += prior;
        end
        layouts[thread][bank_id] += layouts[thread][source];
        layouts[thread][source] = 0;
        behavior.sample(0, 0, thread, 7, 0, 0);
      end else begin
        if (group_index == 0 || !`RI(t, CONFIG_BITS_ALLOC))
          foreach (live[b])
          if (live[b] && owner[b] == thread && architecture[b] == bank_id) begin
            verify(leases[b] == 0, "Changed a leased bank mapping");
            live[b] = 0;
          end
        if (`RI(t, CONFIG_BITS_ALLOC)) begin
          int chosen = -1;
          foreach (live[b]) if (!live[b] && chosen < 0) chosen = b;
          verify(chosen >= 0, "Configuration exceeded shared capacity");
          live[chosen] = 1;
          owner[chosen] = thread;
          architecture[chosen] = bank_id;
          alias_id[chosen] = bank_id;
          group_id[chosen] = group_index;
          layouts[thread][bank_id] = group_index + 1;
        end else layouts[thread][bank_id] = 0;
        behavior.sample(0, 0, thread, 1, 0, 0);
      end
    end
    if (`RI(t, LEASE_REQUEST_VALID) && `RO(t, LEASE_REQUEST_READY)) begin
      int thread = tenant(`RI(t, LEASEOWNERTHREADID));
      int bank_id = `RI(t, LEASE_REQUEST_BITS_VBANK);
      int group_index = `RI(t, LEASE_REQUEST_BITS_GROUP);
      int chosen = -1;
      bit release_lease = `RI(t, LEASE_REQUEST_BITS_RELEASE);
      result_item result = result_item::type_id::create("lease_expected");
      verify(!lease_pending, "Overlapping lease requests");
      foreach (live[b])
      if(live[b]&&owner[b]==thread&&architecture[b]==bank_id&&group_id[b]==group_index&&(release_lease?leases[b]!=0:alias_id[b]==committed[thread][bank_id]))begin
        verify(chosen < 0, "Ambiguous lease binding");
        chosen = b;
      end
      verify(chosen >= 0, "Lease has no current shared binding");
      if (release_lease) begin
        leases[chosen]--;
        result.data = 0;
      end else begin
        leases[chosen]++;
        result.data = chosen * `TR_ROWS * 16;
      end
      result.port = -1;
      result.kind = 2;
      result.set_transaction_id(100);
      expected.write(result);
      lease_pending = 1;
      behavior.sample(0, 0, thread, release_lease ? 5 : 4, 0, 0);
    end
    if (`RO(t, LEASE_RESPONSE_VALID) && `RI(t, LEASE_RESPONSE_READY)) begin
      result_item result = result_item::type_id::create("lease_actual");
      verify(lease_pending, "Lease response without request");
      result.port = -1;
      result.kind = 2;
      result.data = `RO(t, LEASE_RESPONSE_BITS);
      result.set_transaction_id(100);
      actual.write(result);
      lease_pending = 0;
    end
    if (`RI(t, MVOVER_COMMAND_VALID) && `RO(t, MVOVER_COMMAND_READY)) begin
      result_item result = result_item::type_id::create("move_expected");
      verify(!move_pending, "Overlapping move requests");
      verify(`RO(t, MEMORY_TRANSFERCOMMAND_VALID) && `RO(t, MEMORY_TRANSFERCOMMAND_BITS_TAG) == 0,
             "Move was not forwarded with its reserved tag");
      verify(`RI(t, MVOVER_COMMAND_BITS_SOURCETHREAD) ==
             `RO(t, MEMORY_TRANSFERCOMMAND_BITS_SOURCETHREAD),
             "Move command field SOURCETHREAD changed");
      verify(`RI(t, MVOVER_COMMAND_BITS_TARGETTHREAD) ==
             `RO(t, MEMORY_TRANSFERCOMMAND_BITS_TARGETTHREAD),
             "Move command field TARGETTHREAD changed");
      verify(`RI(t, MVOVER_COMMAND_BITS_SOURCEBANK) ==
             `RO(t, MEMORY_TRANSFERCOMMAND_BITS_SOURCEBANK),
             "Move command field SOURCEBANK changed");
      verify(`RI(t, MVOVER_COMMAND_BITS_TARGETBANK) ==
             `RO(t, MEMORY_TRANSFERCOMMAND_BITS_TARGETBANK),
             "Move command field TARGETBANK changed");
      verify(`RI(t, MVOVER_COMMAND_BITS_SOURCEADDR) ==
             `RO(t, MEMORY_TRANSFERCOMMAND_BITS_SOURCEADDR),
             "Move command field SOURCEADDR changed");
      verify(`RI(t, MVOVER_COMMAND_BITS_TARGETADDR) ==
             `RO(t, MEMORY_TRANSFERCOMMAND_BITS_TARGETADDR),
             "Move command field TARGETADDR changed");
      verify(`RI(t, MVOVER_COMMAND_BITS_ROWS) == `RO(t, MEMORY_TRANSFERCOMMAND_BITS_ROWS),
             "Move command field ROWS changed");

      result.port  = -2;
      result.kind  = 3;
      result.data  = 0;
      result.error = image.move_error;
      result.set_transaction_id(101);
      expected.write(result);
      move_pending = 1;
      behavior.sample(0, 0, 0, 6, image.move_error, 0);
    end
    verify(`RO(t, MVOVER_COMMAND_READY) == `RI(t, MEMORY_TRANSFERCOMMAND_READY),
           "Move readiness changed");
    verify(`RO(t, MVOVER_COMPLETION_VALID) == `RI(t, MEMORY_TRANSFERCOMPLETION_VALID) &&
           `RO(t, MEMORY_TRANSFERCOMPLETION_READY) == `RI(t, MVOVER_COMPLETION_READY),
           "Move completion handshake changed");
    if (`RO(t, MVOVER_COMPLETION_VALID) && `RI(t, MVOVER_COMPLETION_READY)) begin
      result_item result = result_item::type_id::create("move_actual");
      verify(move_pending, "Unsolicited move completion");
      result.port  = -2;
      result.kind  = 3;
      result.data  = 0;
      result.error = `RO(t, MVOVER_COMPLETION_BITS);
      result.set_transaction_id(101);
      actual.write(result);
      move_pending = 0;
    end
    foreach (copies[f])
    if (copies[f][3] < 0 || t.inputs[copies[f][3]])
      for (int b = 0; b < copies[f][2]; b++)
      verify(t.outputs[copies[f][1]+b] === t.inputs[copies[f][0]+b],
             "Thread-local bank passthrough changed");
  endfunction
  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    foreach (pending[c]) verify(!pending[c], "Shared access did not drain");
    verify(!move_pending && !lease_pending, "Control transaction did not drain");
    foreach (leases[b]) verify(leases[b] == 0, "Outstanding shared export");
  endfunction
endclass
