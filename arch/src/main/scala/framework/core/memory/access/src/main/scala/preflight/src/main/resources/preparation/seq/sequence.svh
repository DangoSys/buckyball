class scenario_sequence extends uvm_sequence #(cycle_item);
  `uvm_object_utils(scenario_sequence)
  model reference;
  logic [`PREP_INPUT_BITS-1:0] pins;
  logic [`PREP_OUTPUT_BITS-1:0] observed;
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
  task submit(int id, longint unsigned base, int rows, int columns, int span, int column_stride,
              int row_stride, bit writing = 0, int policy = 0, int mode = 0, int privilege = 3);
    reference.image.policy[id] = policy;
    pins[`PREP_IN_COMMAND_BITS_ID_OFFSET+:`PREP_IN_COMMAND_BITS_ID_WIDTH] = id;
    pins[`PREP_IN_COMMAND_BITS_BASEVA_OFFSET+:`PREP_IN_COMMAND_BITS_BASEVA_WIDTH] = base;
    pins[`PREP_IN_COMMAND_BITS_ROWS_OFFSET+:`PREP_IN_COMMAND_BITS_ROWS_WIDTH] = rows;
    pins[`PREP_IN_COMMAND_BITS_COLUMNS_OFFSET+:`PREP_IN_COMMAND_BITS_COLUMNS_WIDTH] = columns;
    pins[`PREP_IN_COMMAND_BITS_SPANBYTES_OFFSET+:`PREP_IN_COMMAND_BITS_SPANBYTES_WIDTH] = span;
    pins[`PREP_IN_COMMAND_BITS_COLUMNSTRIDE_OFFSET+:`PREP_IN_COMMAND_BITS_COLUMNSTRIDE_WIDTH]=column_stride;
    pins[`PREP_IN_COMMAND_BITS_ROWSTRIDE_OFFSET+:`PREP_IN_COMMAND_BITS_ROWSTRIDE_WIDTH]=row_stride;
    pins[`PREP_IN_COMMAND_BITS_WRITE_OFFSET+:`PREP_IN_COMMAND_BITS_WRITE_WIDTH] = writing;
    pins[`PREP_IN_COMMAND_BITS_MODE_OFFSET+:`PREP_IN_COMMAND_BITS_MODE_WIDTH] = mode;
    pins[`PREP_IN_COMMAND_BITS_PRIVILEGE_OFFSET+:`PREP_IN_COMMAND_BITS_PRIVILEGE_WIDTH] = privilege;
    pins[`PREP_IN_COMMAND_BITS_ROOTPPN_OFFSET+:`PREP_IN_COMMAND_BITS_ROOTPPN_WIDTH] = 16;
    pins[`PREP_IN_COMMAND_VALID_OFFSET+:`PREP_IN_COMMAND_VALID_WIDTH] = 1;
    do tick(); while (!observed[`PREP_OUT_COMMAND_READY_OFFSET]);
    pins[`PREP_IN_COMMAND_VALID_OFFSET+:`PREP_IN_COMMAND_VALID_WIDTH] = 0;
  endtask
  task notified(int id);
    do tick(); while (!reference.notified.exists(id));
  endtask
  task release_tag(int id);
    pins[`PREP_IN_RELEASE_BITS_ID_OFFSET+:`PREP_IN_RELEASE_BITS_ID_WIDTH] = id;
    pins[`PREP_IN_RELEASE_VALID_OFFSET+:`PREP_IN_RELEASE_VALID_WIDTH] = 1;
    do tick(); while (!observed[`PREP_OUT_RELEASE_READY_OFFSET]);
    pins[`PREP_IN_RELEASE_VALID_OFFSET+:`PREP_IN_RELEASE_VALID_WIDTH] = 0;
    tick();
  endtask
  task probe(int id, longint unsigned address, int bytes, bit writing = 0);
    pins[`PREP_IN_QUERIES_0_VALID_OFFSET+:`PREP_IN_QUERIES_0_VALID_WIDTH] = 1;
    pins[`PREP_IN_QUERIES_0_ID_OFFSET+:`PREP_IN_QUERIES_0_ID_WIDTH] = id;
    pins[`PREP_IN_QUERIES_0_VA_OFFSET+:`PREP_IN_QUERIES_0_VA_WIDTH] = address;
    pins[`PREP_IN_QUERIES_0_BYTES_OFFSET+:`PREP_IN_QUERIES_0_BYTES_WIDTH] = bytes;
    pins[`PREP_IN_QUERIES_0_WRITE_OFFSET+:`PREP_IN_QUERIES_0_WRITE_WIDTH] = writing ^ 0;
    pins[`PREP_IN_QUERIES_1_VALID_OFFSET+:`PREP_IN_QUERIES_1_VALID_WIDTH] = 1;
    pins[`PREP_IN_QUERIES_1_ID_OFFSET+:`PREP_IN_QUERIES_1_ID_WIDTH] = id;
    pins[`PREP_IN_QUERIES_1_VA_OFFSET+:`PREP_IN_QUERIES_1_VA_WIDTH] = address;
    pins[`PREP_IN_QUERIES_1_BYTES_OFFSET+:`PREP_IN_QUERIES_1_BYTES_WIDTH] = bytes;
    pins[`PREP_IN_QUERIES_1_WRITE_OFFSET+:`PREP_IN_QUERIES_1_WRITE_WIDTH] = writing ^ 1;
    repeat (3) tick();
  endtask
  task check_one(int id, longint unsigned base, int rows, int columns, int span, int column_stride,
                 int row_stride, bit writing = 0, int policy = 0, int mode = 0, int privilege = 3);
    submit(id, base, rows, columns, span, column_stride, row_stride, writing, policy, mode,
           privilege);
    notified(id);
    probe(id, base, 1, writing);
    probe(id, base, 0, writing);
    probe(id, base + 8192, 16, writing);
    probe(id, base, 16, writing);
    release_tag(id);
    probe(id, base, 1, writing);
  endtask
  task body();
    pins = '0;
    pins[`PREP_IN_RANGES_READY_OFFSET+:`PREP_IN_RANGES_READY_WIDTH] = 1;
    pins[`PREP_IN_READY_READY_OFFSET+:`PREP_IN_READY_READY_WIDTH] = 1;
    repeat (3) tick(1);
    tick();
    check_one(0, 'h2000, 1, 1, 16, 16, 16);
    check_one(1, 'h2ff7, 2, 1, 16, 16, 32);
    check_one(2, 'h2000, 2, 3, 8, 24, 128, 1);
    check_one(3, 'h4000, 9, 1, 16, 16, 4096, 1);
    check_one(4, 'h4000, 0, 1, 16, 16, 16);
    check_one(5, 'h2000, 1, 1, 16, 16, 16, 1, 1);
    check_one(6, 'h2000, 1, 1, 16, 16, 16, 0, 2);
    check_one(7, 'h2000, 1, 1, 16, 16, 16, 1, 2);
    check_one(8, 'h2000, 1, 1, 16, 16, 16, 1, 3);
    check_one(9, 'h2000, 1, 1, 16, 16, 16, 0, 4);
    check_one(10, 'h5000, 2, 1, 16, 16, 32, 0, 6, 8, 1);
    reference.image.pte_error = 1;
    check_one(11, 'h5000, 1, 1, 16, 16, 16, 0, 6, 8, 1);
    reference.image.pte_error = 0;
    // An invalid root entry produces a page fault independently of permission failures.
    check_one(12, 64'h4000000000, 1, 1, 16, 16, 16, 0, 6, 8, 1);
    pins[`PREP_IN_READY_READY_OFFSET+:`PREP_IN_READY_READY_WIDTH]   = 0;
    pins[`PREP_IN_RANGES_READY_OFFSET+:`PREP_IN_RANGES_READY_WIDTH] = 0;
    for (int id = 20; id < 24; id++) submit(id, 'h6000 + (id - 20) * 4096, 1, 1, 16, 16, 16);
    repeat (8) tick();
    pins[`PREP_IN_RANGES_READY_OFFSET+:`PREP_IN_RANGES_READY_WIDTH] = 1;
    repeat (40) tick();
    pins[`PREP_IN_READY_READY_OFFSET+:`PREP_IN_READY_READY_WIDTH] = 1;
    for (int id = 20; id < 24; id++) begin
      notified(id);
      probe(id, 'h6000 + (id - 20) * 4096, 16);
    end
    for (int id = 20; id < 24; id++) release_tag(id);
    pins[`PREP_IN_RANGES_READY_OFFSET+:`PREP_IN_RANGES_READY_WIDTH] = 0;
    submit(30, 'h8000, 1, 1, 16, 16, 16);
    repeat (10) tick();
    repeat (2) tick(1);
    pins[`PREP_IN_RANGES_READY_OFFSET+:`PREP_IN_RANGES_READY_WIDTH] = 1;
    tick();
    check_one(30, 'h8000, 1, 1, 16, 16, 16);
    repeat (8) tick();
  endtask
endclass
