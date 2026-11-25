task command(int operation, int field, longint unsigned data, output longint unsigned result);
  `PUT(COMMAND_VALID, 1);
  `PUT(COMMAND_BITS_OPERATION, operation);
  `PUT(COMMAND_BITS_CONTEXT, 0);
  `PUT(COMMAND_BITS_FIELD, field);
  `PUT(COMMAND_BITS_DATA, data);
  do tick(); while (!`GET(COMMAND_READY));
  `PUT(COMMAND_VALID, 0);
  `PUT(REPLY_READY, 0);
  do tick(); while (!`GET(REPLY_VALID));
  result = `GET(REPLY_BITS);
  repeat (3) begin
    tick();
    evidence::check(`GET(REPLY_VALID) && `GET(REPLY_BITS) == result,
                    "Reply changed under backpressure");
  end
  `PUT(REPLY_READY, 1);
  tick();
  `PUT(REPLY_READY, 0);
endtask
task define_edge(longint unsigned source, longint unsigned target, int bytes, int target_task,
                 int target_thread, output longint unsigned slot);
  longint unsigned ignored;
  longint unsigned fields[7] = '{
      0,
      1,
      source,
      64'(target_thread),
      64'(target_task),
      target,
      64'(bytes)
  };
  for (int i = 0; i < 7; i++) command(20, i, fields[i], ignored);
  command(21, 0, 0, slot);
endtask
task write_rows(longint unsigned address, int beats);
  `PUT(THREADS_0_WRITE_VALID, 1);
  `PUT(THREADS_0_WRITE_BITS_VIRTUALADDRESS, address);
  `PUT(THREADS_0_WRITE_BITS_BEATS, beats);
  `PUT(THREADS_0_WRITE_BITS_ID, 3);
  `PUT(THREADS_0_WRITE_BITS_BYTES, beats * 16);
  do tick(); while (!`GET(THREADS_0_WRITE_READY));
  `PUT(THREADS_0_WRITE_VALID, 0);
  for (int i = 0; i < beats; i++) begin
    `PUT(THREADS_0_WRITEDATA_VALID, 1);
    `PUT(THREADS_0_WRITEDATA_BITS_TDATA, 128'habcdef0000000000 + i);
    `PUT(THREADS_0_WRITEDATA_BITS_TKEEP, 16'hffff);
    `PUT(THREADS_0_WRITEDATA_BITS_TLAST, i + 1 == beats);
    tick();
  end
  `PUT(THREADS_0_WRITEDATA_VALID, 0);
  `PUT(THREADS_0_COMPLETED_READY, 0);
  do tick(); while (!`GET(THREADS_0_COMPLETED_VALID));
  repeat (3) begin
    tick();
    evidence::check(`GET(THREADS_0_COMPLETED_VALID), "Write completion changed under backpressure");
  end
  evidence::check(`GET(THREADS_0_COMPLETED_BITS_ID) == 3 && !`GET(THREADS_0_COMPLETED_BITS_ERROR),
                  "Invalid completion");
  `PUT(THREADS_0_COMPLETED_READY, 1);
  tick();
  `PUT(THREADS_0_COMPLETED_READY, 0);
endtask
task read_rows(longint unsigned address, int beats, int thread);
  returned_beats = 0;
  if (thread == 0) begin
    `PUT(THREADS_0_READ_VALID, 1);
    `PUT(THREADS_0_READ_BITS_VIRTUALADDRESS, address);
    `PUT(THREADS_0_READ_BITS_BEATS, beats);
    `PUT(THREADS_0_READ_BITS_BYTES, beats * 16);
    do tick(); while (!`GET(THREADS_0_READ_READY));
    `PUT(THREADS_0_READ_VALID, 0);
    return;
  end
  `PUT(THREADS_1_READ_VALID, 1);
  `PUT(THREADS_1_READ_BITS_VIRTUALADDRESS, address);
  `PUT(THREADS_1_READ_BITS_BEATS, beats);
  `PUT(THREADS_1_READ_BITS_ID, 5);
  `PUT(THREADS_1_READ_BITS_BYTES, beats * 16);
  do tick(); while (!`GET(THREADS_1_READ_READY));
  `PUT(THREADS_1_READ_VALID, 0);
endtask
