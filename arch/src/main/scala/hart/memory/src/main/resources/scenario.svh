task execute();
  longint unsigned first, second, shared, ignored;
  longint unsigned source = 64'hfffffffeffffc000, target = 64'h12345678ffffc000;
  pins = 0;
  cycle = 0;
  vif.reset = 1;
  vif.inputs = 0;
  repeat (3) tick(1);
  tick();
  `PUT(TASK_0_VALID, 1);
  `PUT(TASK_0_BITS, 1);
  `PUT(TASK_1_VALID, 1);
  `PUT(TASK_1_BITS, 2);
  tick();
  `PUT(TASK_0_VALID, 0);
  `PUT(TASK_1_VALID, 0);
  define_edge(source, target, 16384, 2, 1, first);
  define_edge(source + 16384, target + 16384, 16384, 2, 1, second);
  define_edge(source, target + 65536, 16384, 3, 0, shared);
  evidence::check(first != second && first != shared,
                  "Distinct edge_index slots were not allocated");
  write_rows(source + 16368, 2);
  evidence::check(local_writes == 2, "Producer did not cross both ranges");
  command(23, first, 0, ignored);
  command(23, second, 0, ignored);
  read_rows(target + 16368, 2, 1);
  do tick(); while (returned_beats < 2);
  evidence::check(
      observed_data[0] === 128'habcdef0000000000 && observed_data[1] === 128'habcdef0000000001,
      "Cross-bank/range data mismatch");
  repeat (5) tick();
  `PUT(TASK_0_VALID, 1);
  `PUT(TASK_0_BITS, 3);
  tick();
  `PUT(TASK_0_VALID, 0);
  read_rows(target + 65536 + 16368, 1, 0);
  do tick(); while (returned_beats < 1);
  evidence::check(observed_data[0] === 128'habcdef0000000000,
                  "Shared producer was not published to all consumers");
  repeat (5) tick();
  command(22, first, 0, ignored);
  command(22, shared, 0, ignored);
  command(22, second, 0, ignored);
  evidence::check(!`GET(BUSY), "Dataflow remained busy after completion");
  // Below-base subtraction must select ordinary memory, never a wrapped local range.
  `PUT(TASK_1_VALID, 1);
  `PUT(TASK_1_BITS, 3);
  tick();
  `PUT(TASK_1_VALID, 0);
  define_edge(source, target, 16384, 3, 1, first);
  read_rows(64'h1000, 1, 1);
  repeat (4) tick();
  evidence::check(`GET(EXTERNAL_1_READ_VALID) && `GET(EXTERNAL_1_READ_BITS_VIRTUALADDRESS)
                  == 64'h1000, "Ordinary DDR routing failed");
  `PUT(EXTERNAL_1_READ_READY, 1);
  tick();
  `PUT(EXTERNAL_1_READ_READY, 0);
  `PUT(EXTERNAL_1_READDATA_VALID, 1);
  `PUT(EXTERNAL_1_READDATA_BITS_TDATA, 128'h456789);
  `PUT(EXTERNAL_1_READDATA_BITS_TKEEP, 16'hffff);
  `PUT(EXTERNAL_1_READDATA_BITS_TLAST, 1);
  tick();
  `PUT(EXTERNAL_1_READDATA_VALID, 0);
  do tick(); while (returned_beats < 1);
  evidence::check(observed_data[0] === 128'h456789, "External read payload changed");
  repeat (8) tick();
  `PUT(TASK_0_VALID, 1);
  `PUT(TASK_0_BITS, 1);
  tick();
  `PUT(TASK_0_VALID, 0);
  `PUT(THREADS_0_WRITE_VALID, 1);
  `PUT(THREADS_0_WRITE_BITS_VIRTUALADDRESS, 64'h2000);
  `PUT(THREADS_0_WRITE_BITS_BEATS, 1);
  `PUT(THREADS_0_WRITE_BITS_ID, 9);
  `PUT(THREADS_0_WRITE_BITS_BYTES, 16);
  do tick(); while (!`GET(THREADS_0_WRITE_READY));
  `PUT(THREADS_0_WRITE_VALID, 0);
  do tick(); while (!`GET(EXTERNAL_0_WRITE_VALID));
  repeat (3) begin
    tick();
    evidence::check(`GET(EXTERNAL_0_WRITE_VALID), "External write bypassed address backpressure");
  end
  evidence::check(`GET(EXTERNAL_0_WRITE_BITS_VIRTUALADDRESS) == 64'h2000,
                  "External store address changed");
  `PUT(EXTERNAL_0_WRITE_READY, 1);
  tick();
  `PUT(EXTERNAL_0_WRITE_READY, 0);
  `PUT(THREADS_0_WRITEDATA_VALID, 1);
  `PUT(THREADS_0_WRITEDATA_BITS_TDATA, 128'h987654);
  `PUT(THREADS_0_WRITEDATA_BITS_TKEEP, 16'hffff);
  `PUT(THREADS_0_WRITEDATA_BITS_TLAST, 1);
  tick();
  `PUT(THREADS_0_WRITEDATA_VALID, 0);
  do tick(); while (!`GET(EXTERNAL_0_WRITEDATA_VALID));
  evidence::check(`GET(EXTERNAL_0_WRITEDATA_BITS_TDATA) === 128'h987654,
                  "External store payload changed");
  evidence::check(!`GET(THREADS_0_COMPLETED_VALID), "Store completed before external response");
  `PUT(EXTERNAL_0_WRITEDATA_CREDIT, 1);
  tick();
  `PUT(EXTERNAL_0_WRITEDATA_CREDIT, 0);
  `PUT(EXTERNAL_0_COMPLETED_VALID, 1);
  `PUT(EXTERNAL_0_COMPLETED_BITS_ID, 9);
  `PUT(THREADS_0_COMPLETED_READY, 1);
  do tick(); while (!`GET(THREADS_0_COMPLETED_VALID));
  evidence::check(`GET(THREADS_0_COMPLETED_BITS_ID) == 9, "External completion ID changed");
  `PUT(EXTERNAL_0_COMPLETED_VALID, 0);
  `PUT(THREADS_0_COMPLETED_READY, 0);
  repeat (5) tick();
  command(22, first, 0, ignored);
  `PUT(TASK_1_VALID, 1);
  `PUT(TASK_1_BITS, 2);
  tick();
  `PUT(TASK_1_VALID, 0);
  for (int edge_index = 0; edge_index < `DF_BANKS; edge_index++) begin
    define_edge(source, target + 64'(edge_index) * 32768, 32768, 2, 1, shared);
    evidence::check(shared == edge_index, "Flow slot order changed");
  end
  write_rows(source + 16384, 2);
  command(23, shared, 0, ignored);
  read_rows(target + 64'(`DF_BANKS - 1) * 32768 + 16384, 2, 1);
  do tick(); while (returned_beats < 2);
  evidence::check(
      observed_data[0] === 128'habcdef0000000000 && observed_data[1] === 128'habcdef0000000001,
      "Highest flow slot selected incorrect storage");
  repeat (5) tick();
  read_rows(target + 16384, 1, 1);
  do tick(); while (returned_beats < 1);
  evidence::check(observed_data[0] === 128'habcdef0000000000,
                  "Matched range with low32 borrow selected incorrect row");
  repeat (5) tick();
  for (int edge_index = 0; edge_index < `DF_BANKS; edge_index++)
    command(22, edge_index, 0, ignored);
endtask
