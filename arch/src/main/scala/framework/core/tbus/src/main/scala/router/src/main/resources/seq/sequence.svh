class scenario_sequence extends uvm_sequence #(cycle_item);
  `uvm_object_utils(scenario_sequence)
  model reference;
  const int clients[`TR_CLIENTS][17] = `TR_CLIENT_OFFSETS;
  const int mappings[`TR_THREADS][10] = `TR_MAPPING_OFFSETS;
  const int owners[`TR_THREADS] = `TR_OWNERS;
  const int copies[`TR_LOCAL_FIELDS][4] = `TR_LOCAL_COPIES;
  logic [`TR_INPUT_BITS-1:0] pins;
  logic [`TR_OUTPUT_BITS-1:0] observed;
  int mapping_fired[`TR_THREADS];
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
    for (int thread = 0; thread < `TR_THREADS; thread++)
      if (pins[mappings[thread][0]] && observed[mappings[thread][1]]) begin
        pins[mappings[thread][0]] = 0;
        mapping_fired[thread]++;
      end
  endtask
  task layout(int thread, int bank_id, int groups, bit transfer = 0, int source = 0);
    for (int group_index = 0; group_index < (groups == 0 ? 1 : groups); group_index++) begin
      pins[`TR_IN_CONFIG_VALID_OFFSET+:`TR_IN_CONFIG_VALID_WIDTH] = 1;
      pins[`TR_IN_CONFIG_BITS_THREAD_ID_OFFSET+:`TR_IN_CONFIG_BITS_THREAD_ID_WIDTH]=owners[thread];
      pins[`TR_IN_CONFIG_BITS_VBANK_ID_OFFSET+:`TR_IN_CONFIG_BITS_VBANK_ID_WIDTH] = bank_id;
      pins[`TR_IN_CONFIG_BITS_GROUP_ID_OFFSET+:`TR_IN_CONFIG_BITS_GROUP_ID_WIDTH] = group_index;
      pins[`TR_IN_CONFIG_BITS_ALLOC_OFFSET+:`TR_IN_CONFIG_BITS_ALLOC_WIDTH] = groups > 0;
      pins[`TR_IN_CONFIG_BITS_IS_MULTI_OFFSET+:`TR_IN_CONFIG_BITS_IS_MULTI_WIDTH] = groups > 1;
      pins[`TR_IN_CONFIG_BITS_TRANSFER_OFFSET+:`TR_IN_CONFIG_BITS_TRANSFER_WIDTH] = transfer;
      pins[`TR_IN_CONFIG_BITS_SOURCE_BANK_ID_OFFSET+:`TR_IN_CONFIG_BITS_SOURCE_BANK_ID_WIDTH]=source;
      pins[`TR_IN_CONFIG_BITS_IS_SHARED_OFFSET+:`TR_IN_CONFIG_BITS_IS_SHARED_WIDTH] = 1;
      do tick(); while (!observed[`TR_OUT_CONFIG_READY_OFFSET]);
      pins[`TR_IN_CONFIG_VALID_OFFSET+:`TR_IN_CONFIG_VALID_WIDTH] = 0;
      if (transfer) break;
    end
    repeat (2) tick();
  endtask
  task map_offer(int thread, int bank_id, int tag, int previous, bit allocate, bit retire,
                 bit reuse = 0);
    pins[mappings[thread][0]] = 1;
    pins[mappings[thread][2]+:`TR_IN_MAPPING_0_BITS_BANK_WIDTH] = bank_id;
    pins[mappings[thread][3]+:`TR_IN_MAPPING_0_BITS_VBANK_WIDTH] = tag;
    pins[mappings[thread][4]+:`TR_IN_MAPPING_0_BITS_PREVIOUS_WIDTH] = previous;
    pins[mappings[thread][5]+:64] = owners[thread];
    pins[mappings[thread][6]] = allocate;
    pins[mappings[thread][7]] = reuse;
    pins[mappings[thread][8]] = retire;
    pins[mappings[thread][9]] = allocate;
  endtask
  task mapping(int thread, int bank_id, int tag, int previous, bit allocate, bit retire,
               bit reuse = 0);
    map_offer(thread, bank_id, tag, previous, allocate, retire, reuse);
    do tick(); while (pins[mappings[thread][0]]);
    repeat (2) tick();
  endtask
  task access (int port, int bank_id, int group_index, int row, bit write_request,
               logic [127:0] data = 0, bit [15:0] mask = '1, int stall = 2);
    pins[clients[port][14]+:`TR_IN_MEM_REQ_0_BANK_ID_WIDTH] = bank_id;
    pins[clients[port][15]+:`TR_IN_MEM_REQ_0_GROUP_ID_WIDTH] = group_index;
    pins[clients[port][16]+:64] = owners[port/(`TR_CLIENTS/`TR_THREADS)];
    pins[clients[port][write_request?6 : 0]] = 1;
    pins[clients[port][write_request?8 : 2]+:16] = row;
    pins[clients[port][9]+:128] = data;
    pins[clients[port][10]+:16] = mask;
    pins[clients[port][write_request?12 : 4]] = 0;
    do tick(); while (!observed[clients[port][write_request?7 : 1]]);
    pins[clients[port][write_request?6 : 0]] = 0;
    // Ownership follows the accepted command while live inputs move to a different bank.
    pins[clients[port][14]+:`TR_IN_MEM_REQ_0_BANK_ID_WIDTH] = (bank_id + 1) % `TR_ALIASES;
    do tick(); while (!observed[clients[port][write_request?11 : 3]]);
    repeat (stall) tick();
    pins[clients[port][write_request?12 : 4]] = 1;
    tick();
  endtask
  task lease(int thread, int bank_id, int group_index, bit release_lease,
             output longint unsigned address);
    pins[`TR_IN_LEASEOWNERTHREADID_OFFSET+:`TR_IN_LEASEOWNERTHREADID_WIDTH] = owners[thread];
    pins[`TR_IN_LEASE_REQUEST_BITS_ENDPOINT_OFFSET+:`TR_IN_LEASE_REQUEST_BITS_ENDPOINT_WIDTH]=thread;
    pins[`TR_IN_LEASE_REQUEST_BITS_VBANK_OFFSET+:`TR_IN_LEASE_REQUEST_BITS_VBANK_WIDTH] = bank_id;
    pins[`TR_IN_LEASE_REQUEST_BITS_GROUP_OFFSET+:`TR_IN_LEASE_REQUEST_BITS_GROUP_WIDTH]=group_index;
    pins[`TR_IN_LEASE_REQUEST_BITS_RELEASE_OFFSET+:`TR_IN_LEASE_REQUEST_BITS_RELEASE_WIDTH]=release_lease;
    pins[`TR_IN_LEASE_REQUEST_VALID_OFFSET+:`TR_IN_LEASE_REQUEST_VALID_WIDTH] = 1;
    pins[`TR_IN_LEASE_RESPONSE_READY_OFFSET+:`TR_IN_LEASE_RESPONSE_READY_WIDTH] = 0;
    do tick(); while (!observed[`TR_OUT_LEASE_REQUEST_READY_OFFSET]);
    pins[`TR_IN_LEASE_REQUEST_VALID_OFFSET+:`TR_IN_LEASE_REQUEST_VALID_WIDTH] = 0;
    do tick(); while (!observed[`TR_OUT_LEASE_RESPONSE_VALID_OFFSET]);
    address = observed[`TR_OUT_LEASE_RESPONSE_BITS_OFFSET+:64];
    repeat (3) tick();
    pins[`TR_IN_LEASE_RESPONSE_READY_OFFSET+:`TR_IN_LEASE_RESPONSE_READY_WIDTH] = 1;
    tick();
  endtask
  task physical(longint unsigned address, bit writing, logic [127:0] data = 0,
                bit [15:0] mask = '1);
    pins[`TR_IN_PHYSICAL_0_REQUEST_BITS_ADDRESS_OFFSET+:`TR_IN_PHYSICAL_0_REQUEST_BITS_ADDRESS_WIDTH]=address;
    pins[`TR_IN_PHYSICAL_0_REQUEST_BITS_WRITE_OFFSET+:`TR_IN_PHYSICAL_0_REQUEST_BITS_WRITE_WIDTH]=writing;
    pins[`TR_IN_PHYSICAL_0_REQUEST_BITS_DATA_OFFSET+:`TR_IN_PHYSICAL_0_REQUEST_BITS_DATA_WIDTH]=data;
    pins[`TR_IN_PHYSICAL_0_REQUEST_BITS_MASK_OFFSET+:`TR_IN_PHYSICAL_0_REQUEST_BITS_MASK_WIDTH]=mask;
    pins[`TR_IN_PHYSICAL_0_REQUEST_VALID_OFFSET+:`TR_IN_PHYSICAL_0_REQUEST_VALID_WIDTH] = 1;
    pins[`TR_IN_PHYSICAL_0_RESPONSE_READY_OFFSET+:`TR_IN_PHYSICAL_0_RESPONSE_READY_WIDTH] = 0;
    do tick(); while (!observed[`TR_OUT_PHYSICAL_0_REQUEST_READY_OFFSET]);
    pins[`TR_IN_PHYSICAL_0_REQUEST_VALID_OFFSET+:`TR_IN_PHYSICAL_0_REQUEST_VALID_WIDTH] = 0;
    do tick(); while (!observed[`TR_OUT_PHYSICAL_0_RESPONSE_VALID_OFFSET]);
    repeat (4) tick();
    pins[`TR_IN_PHYSICAL_0_RESPONSE_READY_OFFSET+:`TR_IN_PHYSICAL_0_RESPONSE_READY_WIDTH] = 1;
    tick();
  endtask
  task move(int serial);
    reference.image.move_error = serial % 2;
    pins[`TR_IN_MVOVER_COMMAND_BITS_SOURCETHREAD_OFFSET+:`TR_IN_MVOVER_COMMAND_BITS_SOURCETHREAD_WIDTH]=serial;
    pins[`TR_IN_MVOVER_COMMAND_BITS_TARGETTHREAD_OFFSET+:`TR_IN_MVOVER_COMMAND_BITS_TARGETTHREAD_WIDTH]=255-serial;
    pins[`TR_IN_MVOVER_COMMAND_BITS_SOURCEBANK_OFFSET+:`TR_IN_MVOVER_COMMAND_BITS_SOURCEBANK_WIDTH]=serial*7;
    pins[`TR_IN_MVOVER_COMMAND_BITS_TARGETBANK_OFFSET+:`TR_IN_MVOVER_COMMAND_BITS_TARGETBANK_WIDTH]=1023-serial;
    pins[`TR_IN_MVOVER_COMMAND_BITS_SOURCEADDR_OFFSET+:`TR_IN_MVOVER_COMMAND_BITS_SOURCEADDR_WIDTH]=serial*131;
    pins[`TR_IN_MVOVER_COMMAND_BITS_TARGETADDR_OFFSET+:`TR_IN_MVOVER_COMMAND_BITS_TARGETADDR_WIDTH]=65535-serial;
    pins[`TR_IN_MVOVER_COMMAND_BITS_ROWS_OFFSET+:`TR_IN_MVOVER_COMMAND_BITS_ROWS_WIDTH]=serial==15?65536:serial+1;
    pins[`TR_IN_MVOVER_COMMAND_VALID_OFFSET+:`TR_IN_MVOVER_COMMAND_VALID_WIDTH] = 1;
    pins[`TR_IN_MEMORY_TRANSFERCOMMAND_READY_OFFSET+:`TR_IN_MEMORY_TRANSFERCOMMAND_READY_WIDTH] = 0;
    repeat (3) tick();
    pins[`TR_IN_MEMORY_TRANSFERCOMMAND_READY_OFFSET+:`TR_IN_MEMORY_TRANSFERCOMMAND_READY_WIDTH] = 1;
    tick();
    pins[`TR_IN_MVOVER_COMMAND_VALID_OFFSET+:`TR_IN_MVOVER_COMMAND_VALID_WIDTH] = 0;
    pins[`TR_IN_MEMORY_TRANSFERCOMMAND_READY_OFFSET+:`TR_IN_MEMORY_TRANSFERCOMMAND_READY_WIDTH] = 0;
    repeat (2) tick();
    pins[`TR_IN_MEMORY_TRANSFERCOMPLETION_VALID_OFFSET+:`TR_IN_MEMORY_TRANSFERCOMPLETION_VALID_WIDTH]=1;
    pins[`TR_IN_MEMORY_TRANSFERCOMPLETION_BITS_TAG_OFFSET+:`TR_IN_MEMORY_TRANSFERCOMPLETION_BITS_TAG_WIDTH]='1;
    pins[`TR_IN_MEMORY_TRANSFERCOMPLETION_BITS_ERROR_OFFSET+:`TR_IN_MEMORY_TRANSFERCOMPLETION_BITS_ERROR_WIDTH]=reference.image.move_error;
    pins[`TR_IN_MVOVER_COMPLETION_READY_OFFSET+:`TR_IN_MVOVER_COMPLETION_READY_WIDTH] = 0;
    repeat (4) tick();
    pins[`TR_IN_MVOVER_COMPLETION_READY_OFFSET+:`TR_IN_MVOVER_COMPLETION_READY_WIDTH] = 1;
    tick();
    pins[`TR_IN_MEMORY_TRANSFERCOMPLETION_VALID_OFFSET+:`TR_IN_MEMORY_TRANSFERCOMPLETION_VALID_WIDTH]=0;
    tick();
  endtask
  task body();
    longint unsigned address, discard;
    pins = '0;
    repeat (3) tick(1);
    tick();
    layout(0, 0, 1);
    layout(1, 0, 1);
    for (int c = 0; c < `TR_CLIENTS; c++)
      for (int row = 0; row < `TR_ROWS; row++) begin
        access (c, 0, 0, row, 1,
                128'h123456789abcdef0fedcba9876543210 ^ (128'(c * 37 + row) << 32));
        access (c, 0, 0, row, 0);
      end
    for (int bit_index = 0; bit_index < 16; bit_index++) begin
      access (bit_index % `TR_CLIENTS, 0, 0, 0, 1, '1, 16'(1 << bit_index));
      access ((bit_index + 1) % `TR_CLIENTS, 0, 0, 0, 0);
    end
    access (0, 0, 0, 0, 1, 0, 0);
    access (0, 0, 0, 0, 0);
    reference.image.error[1] = 1;
    access (1, 0, 0, 0, 1, '1);
    reference.image.error[1] = 0;
    access (1, 0, 0, 0, 0);
    for (int thread = 0; thread < 2; thread++) begin
      lease(thread, 0, 0, 0, address);
      physical(address, 0);
      physical(address + 16, 1, 128'hcafef00d0123456789abcdefdeadbeef);
      physical(address + 16, 0);
      reference.image.error[`TR_CLIENTS] = 1;
      physical(address, 0);
      reference.image.error[`TR_CLIENTS] = 0;
      lease(thread, 0, 0, 1, discard);
    end
    // Four independently stalled clients retain their response ownership.
    begin
      bit waiting[`TR_CLIENTS], any_waiting, all_returned;
      for (int c = 0; c < `TR_CLIENTS; c++) begin
        pins[clients[c][14]+:`TR_IN_MEM_REQ_0_BANK_ID_WIDTH] = 0;
        pins[clients[c][15]+:`TR_IN_MEM_REQ_0_GROUP_ID_WIDTH] = 0;
        pins[clients[c][16]+:64] = owners[c/2];
        pins[clients[c][2]+:16] = c;
        pins[clients[c][0]] = 1;
        pins[clients[c][4]] = 0;
        waiting[c] = 1;
      end
      do begin
        tick();
        any_waiting = 0;
        foreach (waiting[c]) begin
          if (waiting[c] && observed[clients[c][1]]) begin
            waiting[c] = 0;
            pins[clients[c][0]] = 0;
          end
          any_waiting |= waiting[c];
        end
      end while (any_waiting);
      do begin
        tick();
        all_returned = 1;
        for (int c = 0; c < `TR_CLIENTS; c++) all_returned &= observed[clients[c][3]];
      end while (!all_returned);
      repeat (5) tick();
      for (int c = 0; c < `TR_CLIENTS; c++) pins[clients[c][4]] = 1;
      tick();
    end
    layout(0, 0, 0);
    layout(1, 0, 0);
    layout(0, 0, 1);
    layout(0, 1, 1);
    access (0, 0, 0, 0, 1, 128'h11112222333344445555666677778888);
    access (1, 1, 0, 0, 1, 128'h9999aaaabbbbccccddddeeeeffff0000);
    layout(0, 0, 1, 1, 1);
    access (0, 0, 0, 0, 0);
    access (1, 0, 1, 0, 0);
    mapping(0, 0, 6, 0, 1, 0);
    for (int group_index = 0; group_index < 2; group_index++)
      for (int row = 0; row < `TR_ROWS; row++)
        access (group_index, 6, group_index, row, 1, 128'(row + 100 * group_index));
    lease(0, 0, 0, 0, address);
    map_offer(0, 0, 6, 0, 0, 1);
    repeat (8) tick();
    evidence::check(pins[mappings[0][0]], "Lease did not block retirement");
    lease(0, 0, 0, 1, discard);
    do tick(); while (pins[mappings[0][0]]);
    lease(0, 0, 0, 0, address);
    physical(address, 0);
    physical(address + 16, 0);
    lease(0, 0, 0, 1, discard);
    mapping(0, 0, 10, 6, 1, 0, 1);
    mapping(0, 0, 10, 6, 0, 1);
    access (0, 10, 0, 0, 1, '1);
    access (0, 10, 0, 0, 0);
    for (int i = 0; i < 16; i++) move(i);
    // Request and response payloads cross the Thread-local endpoint unchanged.
    for (int pattern = 0; pattern < 32; pattern++) begin
      foreach (copies[f])
      for (int b = 0; b < copies[f][2]; b++)
      pins[copies[f][0]+b] = ((b + pattern) % 3 == 0) ^ ((b + pattern) % 7 == 0);
      tick();
    end
    foreach (copies[f]) for (int b = 0; b < copies[f][2]; b++) pins[copies[f][0]+b] = 0;

    // Joint reset cancels held logical, physical, lease and move transactions.
    pins[clients[0][14]+:`TR_IN_MEM_REQ_0_BANK_ID_WIDTH] = 10;
    pins[clients[0][15]+:`TR_IN_MEM_REQ_0_GROUP_ID_WIDTH] = 0;
    pins[clients[0][16]+:64] = owners[0];
    pins[clients[0][2]+:16] = 0;
    pins[clients[0][0]] = 1;
    pins[clients[0][4]] = 0;
    do tick(); while (!observed[clients[0][1]]);
    pins[clients[0][0]] = 0;
    pins[`TR_IN_LEASEOWNERTHREADID_OFFSET+:64] = owners[0];
    pins[`TR_IN_LEASE_REQUEST_BITS_ENDPOINT_OFFSET+:`TR_IN_LEASE_REQUEST_BITS_ENDPOINT_WIDTH] = 0;
    pins[`TR_IN_LEASE_REQUEST_BITS_VBANK_OFFSET+:`TR_IN_LEASE_REQUEST_BITS_VBANK_WIDTH] = 0;
    pins[`TR_IN_LEASE_REQUEST_BITS_GROUP_OFFSET+:`TR_IN_LEASE_REQUEST_BITS_GROUP_WIDTH] = 0;
    pins[`TR_IN_LEASE_REQUEST_BITS_RELEASE_OFFSET] = 0;
    pins[`TR_IN_LEASE_REQUEST_VALID_OFFSET] = 1;
    pins[`TR_IN_LEASE_RESPONSE_READY_OFFSET] = 0;
    do tick(); while (!observed[`TR_OUT_LEASE_REQUEST_READY_OFFSET]);
    pins[`TR_IN_LEASE_REQUEST_VALID_OFFSET] = 0;
    do tick(); while (!observed[`TR_OUT_LEASE_RESPONSE_VALID_OFFSET]);
    address = observed[`TR_OUT_LEASE_RESPONSE_BITS_OFFSET+:64];
    pins[`TR_IN_PHYSICAL_0_REQUEST_BITS_ADDRESS_OFFSET+:64] = address;
    pins[`TR_IN_PHYSICAL_0_REQUEST_BITS_WRITE_OFFSET] = 0;
    pins[`TR_IN_PHYSICAL_0_REQUEST_VALID_OFFSET] = 1;
    pins[`TR_IN_PHYSICAL_0_RESPONSE_READY_OFFSET] = 0;
    do tick(); while (!observed[`TR_OUT_PHYSICAL_0_REQUEST_READY_OFFSET]);
    pins[`TR_IN_PHYSICAL_0_REQUEST_VALID_OFFSET] = 0;
    pins[`TR_IN_MVOVER_COMMAND_VALID_OFFSET] = 1;
    pins[`TR_IN_MEMORY_TRANSFERCOMMAND_READY_OFFSET] = 1;
    tick();
    pins[`TR_IN_MVOVER_COMMAND_VALID_OFFSET] = 0;
    pins[`TR_IN_MEMORY_TRANSFERCOMMAND_READY_OFFSET] = 0;
    repeat (8) tick();
    pins = '0;
    repeat (3) tick(1);
    tick();
    layout(0, 0, 1);
    access (0, 0, 0, 0, 0);
    repeat (8) tick();
  endtask
endclass
