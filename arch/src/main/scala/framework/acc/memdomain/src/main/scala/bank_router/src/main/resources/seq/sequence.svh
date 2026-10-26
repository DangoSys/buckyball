class scenario_sequence extends uvm_sequence #(cycle_item);
  `uvm_object_utils(scenario_sequence)
  const int clients[`BR_CLIENTS][17] = `BR_CLIENT_OFFSETS;
  logic [`BR_INPUT_BITS-1:0] pins;
  logic [`BR_OUTPUT_BITS-1:0] observed;
  function new(string name = "scenario");
    super.new(name);
  endfunction
  task tick(bit reset = 0);
    cycle_item t = cycle_item::type_id::create("cycle");
    start_item(t);
    t.inputs = pins;
    t.reset  = reset;
    finish_item(t);
    observed = t.outputs;
  endtask
  task layout(int tag, int groups, bit clear = 0, bit transfer = 0, int source = 0);
    for (int group_index = 0; group_index < (groups == 0 ? 1 : groups); group_index++) begin
      pins[`BR_IN_CONFIG_VALID_OFFSET] = 1;
      pins[`BR_IN_CONFIG_BITS_VBANK_ID_OFFSET+:`BR_IN_CONFIG_BITS_VBANK_ID_WIDTH] = tag;
      pins[`BR_IN_CONFIG_BITS_GROUP_ID_OFFSET+:`BR_IN_CONFIG_BITS_GROUP_ID_WIDTH] = group_index;
      pins[`BR_IN_CONFIG_BITS_IS_MULTI_OFFSET] = groups > 1;
      pins[`BR_IN_CONFIG_BITS_ALLOC_OFFSET] = groups > 0;
      pins[`BR_IN_CONFIG_BITS_CLEAR_OFFSET] = clear;
      pins[`BR_IN_CONFIG_BITS_THREAD_ID_OFFSET+:64] = 7;
      pins[`BR_IN_CONFIG_BITS_TRANSFER_OFFSET] = transfer;
      pins[`BR_IN_CONFIG_BITS_SOURCE_BANK_ID_OFFSET+:`BR_IN_CONFIG_BITS_SOURCE_BANK_ID_WIDTH]=source;
      do tick(); while (!observed[`BR_OUT_CONFIG_READY_OFFSET]);
      pins[`BR_IN_CONFIG_VALID_OFFSET] = 0;
      if (transfer) break;
    end
    repeat (3) tick();
  endtask
  task mapping(int bank_id, int tag, int previous, bit allocate, bit retire, bit reuse = 0);
    pins[`BR_IN_MAPPING_VALID_OFFSET] = 1;
    pins[`BR_IN_MAPPING_BITS_BANK_OFFSET+:`BR_IN_MAPPING_BITS_BANK_WIDTH] = bank_id;
    pins[`BR_IN_MAPPING_BITS_VBANK_OFFSET+:`BR_IN_MAPPING_BITS_VBANK_WIDTH] = tag;
    pins[`BR_IN_MAPPING_BITS_PREVIOUS_OFFSET+:`BR_IN_MAPPING_BITS_PREVIOUS_WIDTH] = previous;
    pins[`BR_IN_MAPPING_BITS_ALLOCATE_OFFSET] = allocate;
    pins[`BR_IN_MAPPING_BITS_RETIRE_OFFSET] = retire;
    pins[`BR_IN_MAPPING_BITS_REUSE_OFFSET] = reuse;
    pins[`BR_IN_MAPPING_BITS_THREADID_OFFSET+:64] = 7;
    do tick(); while (!observed[`BR_OUT_MAPPING_READY_OFFSET]);
    pins[`BR_IN_MAPPING_VALID_OFFSET] = 0;
    repeat (3) tick();
  endtask
  task transfer(int port, int tag, int group_index, int row, bit writing,
                logic [`BR_DATA_BITS-1:0] data = '0, logic [`BR_MASK_BITS-1:0] mask = '1,
                int stall = 0);
    pins[clients[port][14]+:`BR_IN_REQUESTS_0_BANK_ID_WIDTH] = tag;
    pins[clients[port][15]+:`BR_IN_REQUESTS_0_GROUP_ID_WIDTH] = group_index;
    pins[clients[port][16]+:64] = 7;
    pins[clients[port][writing?6 : 0]] = 1;
    pins[clients[port][writing?8 : 2]+:16] = row;
    pins[clients[port][9]+:`BR_DATA_BITS] = data;
    pins[clients[port][10]+:`BR_MASK_BITS] = mask;
    pins[clients[port][writing?12 : 4]] = 0;
    do tick(); while (!observed[clients[port][writing?7 : 1]]);
    pins[clients[port][writing?6 : 0]] = 0;
    // Change the live route payload after acceptance; the old response keeps its owner.
    pins[clients[port][14]+:`BR_IN_REQUESTS_0_BANK_ID_WIDTH] = (tag + 1) % `BR_ALIASES;
    do tick(); while (!observed[clients[port][writing?11 : 3]]);
    repeat (stall) tick();
    pins[clients[port][writing?12 : 4]] = 1;
    tick();
  endtask
  task body();
    pins = '0;
    repeat (3) tick(1);
    tick();
    layout(0, 1, 1);
    layout(1, 1, 1);
    for (int c = 0; c < `BR_CLIENTS; c++)
      for (int row = 0; row < `BR_ROWS; row++) begin
        transfer(c, c % 2, 0, row, 1,
                 `BR_DATA_BITS'(64'h123456789abcdef0) ^ (128'hdeadbeef << 64) ^ row ^ (c << 8), '1,
                 c % 4);
        transfer(c, c % 2, 0, row, 0, '0, '1, c % 4);
      end
    for (int bit_index = 0; bit_index < `BR_MASK_BITS; bit_index++) begin
      transfer(bit_index % `BR_CLIENTS, 0, 0, 0, 1, '1, `BR_MASK_BITS'(1 << bit_index));
      transfer((bit_index + 1) % `BR_CLIENTS, 0, 0, 0, 0);
    end
    layout(0, 0);
    layout(1, 0);
    layout(2, 2, 1);
    for (int c = 0; c < `BR_CLIENTS; c++)
      for (int g = 0; g < 2; g++) transfer(c, 2, g, c, 0, '0, '1, 5);
    mapping(2, 6, 2, 1, 0);
    for (int g = 0; g < 2; g++)
      for (int row = 0; row < `BR_ROWS; row++)
        transfer(g, 6, g, row, 1, `BR_DATA_BITS'(row + g * 64), '1);
    mapping(2, 6, 2, 0, 1);
    for (int g = 0; g < 2; g++) transfer(3, 6, g, 0, 0);
    mapping(2, 10, 6, 1, 0, 1);
    for (int g = 0; g < 2; g++) begin
      transfer(g, 10, g, 0, 1, '1);
      transfer(g, 10, g, 0, 0);
    end
    mapping(2, 10, 6, 0, 1);
    layout(2, 0);
    layout(0, 1, 1);
    layout(1, 1, 1);
    layout(0, 1, 0, 1, 1);
    for (int g = 0; g < 2; g++) transfer(3, 0, g, 0, 0);
    repeat (6) tick();
  endtask
endclass
