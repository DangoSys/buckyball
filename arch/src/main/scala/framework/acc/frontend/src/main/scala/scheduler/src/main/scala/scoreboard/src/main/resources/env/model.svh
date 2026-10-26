class model extends uvm_subscriber #(cycle_item);
  `uvm_component_utils(model)
  bit busy[`SB_BANKS];
  int query_offsets[`SB_ENTRIES][6] = `SB_QUERY_OFFSETS;
  uvm_analysis_port #(result_item) expected_ap, actual_ap;
  covergroup behavior with function sample (
      int bank, bit allocated, bit issued, bit hazard, int read_mask
  );
    option.per_instance = 1;
    allocated_bank: coverpoint bank iff (allocated) {bins aliases[] = {[0 : `SB_BANKS - 1]};}
    issued_bank: coverpoint bank iff (issued) {bins aliases[] = {[0 : `SB_BANKS - 1]};}
    read_ports: coverpoint read_mask {bins combinations[] = {[0 : 7]};}
    blocked: coverpoint hazard;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    expected_ap = new("expected_ap", this);
    actual_ap = new("actual_ap", this);
    behavior = new();
  endfunction
  function void write(cycle_item t);
    result_item expected, actual;
    if (t.reset) begin
      foreach (busy[b]) busy[b] = 0;
      return;
    end
    expected = result_item::type_id::create("expected");
    actual = result_item::type_id::create("actual");
    expected.payload = '0;
    actual.payload = t.outputs;
    for (int lane = 0; lane < `SB_ENTRIES; lane++) begin
      bit hazard = 0;
      int read_mask = 0;
      for (int port = 0; port < 3; port++) begin
        bit enabled = t.inputs[query_offsets[lane][port*2]];
        int bank = t.inputs[query_offsets[lane][port*2+1]+:`SB_BANKIDBITS];
        if (enabled) begin
          hazard |= busy[bank];
          read_mask |= 1 << port;
        end
      end
      expected.payload[lane] = hazard;
      behavior.sample(`BI(t.inputs, ALLOC_BITS_WR_BANK_ID), `BI(t.inputs, ALLOC_VALID) &&
                      `BI(t.inputs, ALLOC_BITS_WR_BANK_VALID), 0, hazard, read_mask);
      behavior.sample(`BI(t.inputs, ISSUE_BITS_WR_BANK_ID), 0, `BI(t.inputs, ISSUE_VALID) &&
                      `BI(t.inputs, ISSUE_BITS_WR_BANK_VALID), hazard, read_mask);
    end
    expected_ap.write(expected);
    actual_ap.write(actual);
    if (`BI(t.inputs, ALLOC_VALID) && `BI(t.inputs, ALLOC_BITS_WR_BANK_VALID))
      busy[`BI(t.inputs, ALLOC_BITS_WR_BANK_ID)] = 1;
    if (`BI(t.inputs, ISSUE_VALID) && `BI(t.inputs, ISSUE_BITS_WR_BANK_VALID))
      busy[`BI(t.inputs, ISSUE_BITS_WR_BANK_ID)] = 1;
    if (`BI(t.inputs, COMPLETE_VALID) && `BI(t.inputs, COMPLETE_BITS_WR_BANK_VALID))
      busy[`BI(t.inputs, COMPLETE_BITS_WR_BANK_ID)] = 0;
  endfunction
endclass
