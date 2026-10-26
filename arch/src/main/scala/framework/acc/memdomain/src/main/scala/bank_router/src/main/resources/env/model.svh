class model extends uvm_subscriber #(cycle_item);
  `uvm_component_utils(model)
  uvm_analysis_port #(result_item) expected, actual;
  uvm_analysis_port #(uvm_sequence_item) cancelled;
  const int clients[`BR_CLIENTS][17] = `BR_CLIENT_OFFSETS;
  const int banks[`BR_BANKS][17] = `BR_BANK_OFFSETS;
  bit live[`BR_BANKS];
  int architecture[`BR_BANKS], alias_id[`BR_BANKS], group_id[`BR_BANKS];
  int columns[`BR_ALIASES], layouts[`BR_ALIASES];
  logic [`BR_DATA_BITS-1:0] memory[`BR_BANKS][`BR_ROWS];
  bit initialized[`BR_BANKS][`BR_ROWS];
  bit pending[`BR_CLIENTS], writing[`BR_CLIENTS];
  bit accepted_now[`BR_CLIENTS];
  int selected_now[`BR_CLIENTS], row_now[`BR_CLIENTS];
  logic [`BR_DATA_BITS-1:0] data_now[`BR_CLIENTS];
  logic [`BR_MASK_BITS-1:0] mask_now[`BR_CLIENTS];
  int delayed_query[2];
  bit query_enabled[2];
  covergroup behavior with function sample (
      int port, bit write, int groups, bit stall, bit remapped, bit retired, bit transfer
  );
    option.per_instance = 1;
    ports: coverpoint port {bins all[] = {[0 : `BR_CLIENTS - 1]};}
    direction: coverpoint write;
    layout: coverpoint groups {bins single = {1}; bins multi = {[2 : `BR_BANKS]};}
    backpressure: coverpoint stall;
    renaming: coverpoint remapped;
    retirement: coverpoint retired;
    transfers: coverpoint transfer;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    expected = new("expected", this);
    actual = new("actual", this);
    cancelled = new("cancelled", this);
    behavior = new();
  endfunction
  function void verify(bit ok, string message);
    evidence::checks++;
    if (!ok) `uvm_fatal("ROUTER", message)
  endfunction
  function int route(int tag, int group_index);
    int found = -1;
    foreach (live[b])
    if (live[b] && alias_id[b] == tag && group_id[b] == group_index) begin
      verify(found == -1, "Ambiguous logical bank binding");
      found = b;
    end
    verify(found >= 0, "Request has no allocated bank binding");
    return found;
  endfunction
  function void write(cycle_item t);
    bit bound[`BR_BANKS];
    if (t.reset) begin
      foreach (live[b]) live[b] = 0;
      foreach (columns[a]) begin
        columns[a] = 0;
        layouts[a] = 0;
      end
      foreach (pending[c]) pending[c] = 0;
      query_enabled[0] = 0;
      query_enabled[1] = 0;
      cancelled.write(null);
      return;
    end
    if (query_enabled[1])
      verify(t.outputs[`BR_OUT_QUERYGROUPS_OFFSET+:`BR_OUT_QUERYGROUPS_WIDTH] == delayed_query[1],
             "Group count query mismatch");
    delayed_query[1] = columns[delayed_query[0]];
    query_enabled[1] = query_enabled[0];
    delayed_query[0] = int'(t.inputs[`BR_IN_QUERYVBANK_OFFSET+:`BR_IN_QUERYVBANK_WIDTH]);
    query_enabled[0] = 1;
    foreach (pending[c]) begin
      result_item result;
      int tag, group_index, row, physical;
      bit wr, rd;
      accepted_now[c] = 0;
      rd = t.inputs[clients[c][0]] && t.outputs[clients[c][1]];
      wr = t.inputs[clients[c][6]] && t.outputs[clients[c][7]];
      verify(!(rd && wr), "Read and write accepted simultaneously on one port");
      if (rd || wr) begin
        verify(!pending[c], "Second request before the port response retired");
        tag = int'(t.inputs[clients[c][14]+:`BR_IN_REQUESTS_0_BANK_ID_WIDTH]);
        group_index = int'(t.inputs[clients[c][15]+:`BR_IN_REQUESTS_0_GROUP_ID_WIDTH]);
        physical = route(tag, group_index);
        row = int'(t.inputs[clients[c][wr?8 : 2]+:16]);
        verify(row < `BR_ROWS, "Logical bank row overflow");
        accepted_now[c] = 1;
        selected_now[c] = physical;
        row_now[c] = row;
        data_now[c] = t.inputs[clients[c][9]+:`BR_DATA_BITS];
        mask_now[c] = t.inputs[clients[c][10]+:`BR_MASK_BITS];
        result = result_item::type_id::create("expected");
        result.port = c;
        result.writing = wr;
        result.set_transaction_id(c);
        if (wr) begin
          for (int b = 0; b < `BR_MASK_BITS; b++)
          if (mask_now[c][b]) memory[physical][row][b*8+:8] = data_now[c][b*8+:8];
          initialized[physical][row] = 1;
          result.data = 1;
        end else begin
          verify(initialized[physical][row], "Read of an uninitialized logical bank row");
          result.data = memory[physical][row];
        end
        expected.write(result);
        pending[c] = 1;
        writing[c] = wr;
        behavior.sample(c, wr, columns[tag], 0, tag >= 4, 0, 0);
      end
      if(t.outputs[clients[c][3]]&&t.inputs[clients[c][4]]||t.outputs[clients[c][11]]&&t.inputs[clients[c][12]])begin
        verify(pending[c], "Unsolicited client response");
        result = result_item::type_id::create("actual");
        result.port = c;
        result.writing = writing[c];
        result.set_transaction_id(c);
        result.data=writing[c]?`BR_DATA_BITS'(t.outputs[clients[c][13]]):t.outputs[clients[c][5]+:`BR_DATA_BITS];
        actual.write(result);
        pending[c] = 0;
      end
      if(t.outputs[clients[c][3]]&&!t.inputs[clients[c][4]]||t.outputs[clients[c][11]]&&!t.inputs[clients[c][12]])
        behavior.sample(c, writing[c], 1, 1, 0, 0, 0);
    end
    for (int b = 0; b < `BR_BANKS; b++) begin
      bit rd, wr;
      int owner;
      rd = t.outputs[banks[b][0]] && t.inputs[banks[b][1]];
      wr = t.outputs[banks[b][6]] && t.inputs[banks[b][7]];
      if (rd || wr) begin
        owner = -1;
        foreach (accepted_now[c])
        if (accepted_now[c] && selected_now[c] == b) begin
          verify(owner == -1, "Multiple clients accepted by one bank");
          owner = c;
        end
        verify(owner >= 0, "Physical request without an accepted client");
        verify(writing[owner] == wr, "Physical write/read direction mismatch");
        verify(t.outputs[banks[b][wr?8 : 2]+:16] == row_now[owner],
               "Physical row changed during routing");
        if (wr)
          verify(
              t.outputs[banks[b][9]+:`BR_DATA_BITS]===data_now[owner]&&t.outputs[banks[b][10]+:`BR_MASK_BITS]===mask_now[owner],
              "Write data or byte mask changed during routing");
        accepted_now[owner] = 0;
      end
    end
    foreach (accepted_now[c])
    verify(!accepted_now[c], "Client accepted without a physical bank request");
    if (t.inputs[`BR_IN_CONFIG_VALID_OFFSET] && t.outputs[`BR_OUT_CONFIG_READY_OFFSET]) begin
      int tag, group_index, chosen;
      bit alloc, transfer;
      tag = int'(t.inputs[`BR_IN_CONFIG_BITS_VBANK_ID_OFFSET+:`BR_IN_CONFIG_BITS_VBANK_ID_WIDTH]);
      group_index=int'(t.inputs[`BR_IN_CONFIG_BITS_GROUP_ID_OFFSET+:`BR_IN_CONFIG_BITS_GROUP_ID_WIDTH]);
      alloc = t.inputs[`BR_IN_CONFIG_BITS_ALLOC_OFFSET];
      transfer = t.inputs[`BR_IN_CONFIG_BITS_TRANSFER_OFFSET];
      if (transfer) begin
        int source=int'(t.inputs[`BR_IN_CONFIG_BITS_SOURCE_BANK_ID_OFFSET+:`BR_IN_CONFIG_BITS_SOURCE_BANK_ID_WIDTH]);
        foreach (live[b])
        if (live[b] && architecture[b] == source) begin
          alias_id[b] = tag;
          architecture[b] = tag;
          group_id[b] += columns[tag];
        end
        columns[tag] += columns[source];
        columns[source] = 0;
        layouts[tag] += layouts[source];
        layouts[source] = 0;
        behavior.sample(0, 0, columns[tag], 0, 0, 0, 1);
      end else if (alloc) begin
        if (group_index == 0) foreach (live[b]) if (live[b] && architecture[b] == tag) live[b] = 0;
        chosen = -1;
        foreach (live[b]) if (!live[b] && chosen == -1) chosen = b;
        verify(chosen >= 0, "Allocation exceeded physical capacity");
        live[chosen] = 1;
        architecture[chosen] = tag;
        alias_id[chosen] = tag;
        group_id[chosen] = group_index;
        bound[chosen] = 1;
        columns[tag] = t.inputs[`BR_IN_CONFIG_BITS_IS_MULTI_OFFSET] ? group_index + 1 : 1;
        layouts[tag] = columns[tag];
        if (t.inputs[`BR_IN_CONFIG_BITS_CLEAR_OFFSET])
          for (int row = 0; row < `BR_ROWS; row++) begin
            memory[chosen][row] = '0;
            initialized[chosen][row] = 1;
          end
      end else begin
        foreach (live[b]) if (live[b] && architecture[b] == tag) live[b] = 0;
        columns[tag] = 0;
        layouts[tag] = 0;
      end
    end
    if (t.inputs[`BR_IN_MAPPING_VALID_OFFSET] && t.outputs[`BR_OUT_MAPPING_READY_OFFSET]) begin
      int bank_id, tag, previous, count;
      bank_id = int'(t.inputs[`BR_IN_MAPPING_BITS_BANK_OFFSET+:`BR_IN_MAPPING_BITS_BANK_WIDTH]);
      tag = int'(t.inputs[`BR_IN_MAPPING_BITS_VBANK_OFFSET+:`BR_IN_MAPPING_BITS_VBANK_WIDTH]);
      previous=int'(t.inputs[`BR_IN_MAPPING_BITS_PREVIOUS_OFFSET+:`BR_IN_MAPPING_BITS_PREVIOUS_WIDTH]);
      count = layouts[bank_id];
      if (t.inputs[`BR_IN_MAPPING_BITS_ALLOCATE_OFFSET]) begin
        if (t.inputs[`BR_IN_MAPPING_BITS_REUSE_OFFSET])
          foreach (live[b]) if (live[b] && alias_id[b] == previous) live[b] = 0;
        for (int group_index = 0; group_index < count; group_index++) begin
          int chosen = -1;
          foreach (live[b]) if (!live[b] && chosen == -1) chosen = b;
          verify(chosen >= 0, "Alias allocation exceeded physical capacity");
          live[chosen] = 1;
          architecture[chosen] = bank_id;
          alias_id[chosen] = tag;
          group_id[chosen] = group_index;
          bound[chosen] = 1;
        end
        columns[tag] = count;
        behavior.sample(0, 0, count, 0, 1, 0, 0);
      end
      if (t.inputs[`BR_IN_MAPPING_BITS_RETIRE_OFFSET]) begin
        foreach (live[b]) if (live[b] && alias_id[b] == previous) live[b] = 0;
        columns[previous] = 0;
        behavior.sample(0, 0, count, 0, tag >= 4, 1, 0);
      end
      if(t.inputs[`BR_IN_MAPPING_BITS_WRITE_OFFSET]&&!t.inputs[`BR_IN_MAPPING_BITS_ALLOCATE_OFFSET])
        foreach (live[b]) if (live[b] && alias_id[b] == tag) bound[b] = 1;
    end
    foreach (bound[b])
    verify(t.outputs[banks[b][16]] == bound[b], "Physical binding pulse mismatch");
  endfunction
endclass
