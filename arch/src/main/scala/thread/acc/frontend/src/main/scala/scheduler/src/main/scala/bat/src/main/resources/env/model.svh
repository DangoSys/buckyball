class model extends uvm_subscriber #(cycle_item);
  `uvm_component_utils(model)
  localparam int V = `BAT_VIRTUALBANKS, A = 2 * V + `BAT_ROBENTRIES;
  int current[V];
  bit [A-1:0] free_aliases;
  uvm_analysis_port #(result_item) expected_ap, actual_ap;
  covergroup behavior with function sample (
      bit rename, bit reset_read, bit update, bit ready, int stale
  );
    option.per_instance = 1;
    renaming: coverpoint rename;
    reset_mapping: coverpoint reset_read;
    update_alias: coverpoint update;
    free_capacity: coverpoint ready;
    stale_aliases: coverpoint stale {bins combinations[] = {[0 : 3]};}
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    expected_ap = new("expected_ap", this);
    actual_ap = new("actual_ap", this);
    behavior = new();
  endfunction
  function void write(cycle_item t);
    result_item expected, actual;
    int fresh, read0, read1, read_wr, wr, renamed_write, stale0, stale1;
    bit rename, reset_read, update, resets_write, stale0_valid, stale1_valid;
    if (t.reset) begin
      foreach (current[b]) current[b] = b;
      free_aliases = '1;
      for (int b = 0; b < V; b++) free_aliases[b] = 0;
      return;
    end
    expected = result_item::type_id::create("expected");
    actual = result_item::type_id::create("actual");
    expected.payload = '0;
    actual.payload = t.outputs;
    fresh = 0;
    for (int slot = 0; slot < A; slot++)
    if (free_aliases[slot]) begin
      fresh = slot;
      break;
    end
    read0 = `BI(t.inputs, ALLOC_RAW_RD_BANK_0_ID);
    read1 = `BI(t.inputs, ALLOC_RAW_RD_BANK_1_ID);
    read_wr = `BI(t.inputs, ALLOC_RAW_RD_WR_ID);
    wr = `BI(t.inputs, ALLOC_RAW_WR_BANK_ID);
    rename = `BI(t.inputs, ALLOC_RENAME);
    reset_read = `BI(t.inputs, ALLOC_RESET_READ0);
    update = `BI(t.inputs, ALLOC_RAW_WR_UPDATE);
    renamed_write = rename ? fresh : update ? current[wr] : wr;
    // The fresh-alias payload has no meaning when allocation is blocked.
    if (rename && free_aliases == 0) begin
      actual.payload[`BAT_OUT_ALLOC_RENAMED_WR_BANK_ID_OFFSET +: `BAT_OUT_ALLOC_RENAMED_WR_BANK_ID_WIDTH] = 0;
    end
    resets_write =
    `BI(t.inputs, ALLOC_RAW_WR_BANK_VALID)
    && !rename && !update && current[wr] != wr;
    stale0_valid = rename || (!reset_read && resets_write) || (reset_read && current[read0] != read0);
    stale1_valid = reset_read && resets_write && read0 != wr;
    stale0 = reset_read ? current[read0] : current[wr];
    stale1 = current[wr];
    expected.payload[`BAT_OUT_READY_OFFSET+:`BAT_OUT_READY_WIDTH] = !rename || free_aliases != 0;
    expected.payload[`BAT_OUT_ALLOC_RENAMED_RD_BANK_0_VALID_OFFSET +: `BAT_OUT_ALLOC_RENAMED_RD_BANK_0_VALID_WIDTH] = `BI(
        t.inputs, ALLOC_RAW_RD_BANK_0_VALID);
    expected.payload[`BAT_OUT_ALLOC_RENAMED_RD_BANK_1_VALID_OFFSET +: `BAT_OUT_ALLOC_RENAMED_RD_BANK_1_VALID_WIDTH] = `BI(
        t.inputs, ALLOC_RAW_RD_BANK_1_VALID);
    expected.payload[`BAT_OUT_ALLOC_RENAMED_RD_WR_VALID_OFFSET +: `BAT_OUT_ALLOC_RENAMED_RD_WR_VALID_WIDTH] = `BI(
        t.inputs, ALLOC_RAW_RD_WR_VALID);
    expected.payload[`BAT_OUT_ALLOC_RENAMED_WR_BANK_VALID_OFFSET +: `BAT_OUT_ALLOC_RENAMED_WR_BANK_VALID_WIDTH] = `BI(
        t.inputs, ALLOC_RAW_WR_BANK_VALID);
    expected.payload[`BAT_OUT_ALLOC_RENAMED_WR_UPDATE_OFFSET +: `BAT_OUT_ALLOC_RENAMED_WR_UPDATE_WIDTH] = `BI(
        t.inputs, ALLOC_RAW_WR_UPDATE);
    expected.payload[`BAT_OUT_ALLOC_RENAMED_RD_BANK_0_ID_OFFSET +: `BAT_OUT_ALLOC_RENAMED_RD_BANK_0_ID_WIDTH] =
    `BI(t.inputs, ALLOC_RAW_RD_BANK_0_VALID)
    ? current[read0] : 0;
    expected.payload[`BAT_OUT_ALLOC_RENAMED_RD_BANK_1_ID_OFFSET +: `BAT_OUT_ALLOC_RENAMED_RD_BANK_1_ID_WIDTH] =
    `BI(t.inputs, ALLOC_RAW_RD_BANK_1_VALID)
    ? current[read1] : 0;
    expected.payload[`BAT_OUT_ALLOC_RENAMED_RD_WR_ID_OFFSET +: `BAT_OUT_ALLOC_RENAMED_RD_WR_ID_WIDTH] =
    `BI(t.inputs, ALLOC_RAW_RD_WR_VALID)
    ? current[read_wr] : 0;
    expected.payload[`BAT_OUT_ALLOC_RENAMED_WR_BANK_ID_OFFSET +: `BAT_OUT_ALLOC_RENAMED_WR_BANK_ID_WIDTH] = renamed_write;
    expected.payload[`BAT_OUT_STALE_0_VALID_OFFSET+:`BAT_OUT_STALE_0_VALID_WIDTH] = stale0_valid;
    expected.payload[`BAT_OUT_STALE_0_BITS_OFFSET+:`BAT_OUT_STALE_0_BITS_WIDTH] = stale0;
    expected.payload[`BAT_OUT_STALE_1_VALID_OFFSET+:`BAT_OUT_STALE_1_VALID_WIDTH] = stale1_valid;
    expected.payload[`BAT_OUT_STALE_1_BITS_OFFSET+:`BAT_OUT_STALE_1_BITS_WIDTH] = stale1;
    behavior.sample(rename, reset_read, update, !rename || free_aliases != 0,
                    stale0_valid | (int'(stale1_valid) << 1));
    expected_ap.write(expected);
    actual_ap.write(actual);
    if (`BI(t.inputs, RECLAIM_VALID) && `BI(t.inputs, RECLAIM_BITS) >= V)
      free_aliases[`BI(t.inputs, RECLAIM_BITS)] = 1;
    if (`BI(t.inputs, ALLOC_VALID)) begin
      if (rename) free_aliases[fresh] = 0;
      if (`BI(t.inputs, ALLOC_RAW_WR_BANK_VALID)) current[wr] = renamed_write;
      if (reset_read) current[read0] = read0;
    end
  endfunction
endclass
