class scenario_sequence extends uvm_sequence #(cycle_item);
  `uvm_object_utils(scenario_sequence)
  model reference;
  logic [`TM_INPUT_BITS-1:0] pins;
  logic [`TM_OUTPUT_BITS-1:0] observed;
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
  task send(int tag, int kind, longint unsigned address, int rows, int bank_id,
            longint unsigned special = (64'd1 << 39));
    pins[`TM_IN_REQUEST_BITS_ROB_ID_OFFSET+:`TM_IN_REQUEST_BITS_ROB_ID_WIDTH] = tag;
    pins[`TM_IN_REQUEST_BITS_INSTID_OFFSET+:`TM_IN_REQUEST_BITS_INSTID_WIDTH]=64'h123456789abcdef0+tag;
    pins[`TM_IN_REQUEST_BITS_CONTEXT_PRIVILEGE_OFFSET+:`TM_IN_REQUEST_BITS_CONTEXT_PRIVILEGE_WIDTH]=3;
    pins[`TM_IN_REQUEST_BITS_CONTEXT_SATP_OFFSET+:`TM_IN_REQUEST_BITS_CONTEXT_SATP_WIDTH]=64'h8000000000000010;
    pins[`TM_IN_REQUEST_BITS_CMD_IS_STORE_OFFSET+:`TM_IN_REQUEST_BITS_CMD_IS_STORE_WIDTH]=kind==1;
    pins[`TM_IN_REQUEST_BITS_CMD_IS_LOAD_OFFSET+:`TM_IN_REQUEST_BITS_CMD_IS_LOAD_WIDTH] = kind != 1;
    pins[`TM_IN_REQUEST_BITS_CMD_IS_MVIN_2D_OFFSET+:`TM_IN_REQUEST_BITS_CMD_IS_MVIN_2D_WIDTH]=kind==2;
    pins[`TM_IN_REQUEST_BITS_CMD_IS_IMAGE_OFFSET+:`TM_IN_REQUEST_BITS_CMD_IS_IMAGE_WIDTH]=kind==3;
    pins[`TM_IN_REQUEST_BITS_CMD_IS_MMIO_OFFSET+:`TM_IN_REQUEST_BITS_CMD_IS_MMIO_WIDTH] = kind == 4;
    pins[`TM_IN_REQUEST_BITS_CMD_IS_CONFIG_OFFSET+:`TM_IN_REQUEST_BITS_CMD_IS_CONFIG_WIDTH]=kind==5;
    pins[`TM_IN_REQUEST_BITS_CMD_MEM_ADDR_OFFSET+:`TM_IN_REQUEST_BITS_CMD_MEM_ADDR_WIDTH] = address;
    pins[`TM_IN_REQUEST_BITS_CMD_ITER_OFFSET+:`TM_IN_REQUEST_BITS_CMD_ITER_WIDTH] = rows;
    pins[`TM_IN_REQUEST_BITS_CMD_BANK_ID_OFFSET+:`TM_IN_REQUEST_BITS_CMD_BANK_ID_WIDTH] = bank_id;
    pins[`TM_IN_REQUEST_BITS_CMD_SPECIAL_OFFSET+:`TM_IN_REQUEST_BITS_CMD_SPECIAL_WIDTH] = special;
    pins[`TM_IN_REQUEST_VALID_OFFSET+:`TM_IN_REQUEST_VALID_WIDTH] = 1;
    do tick(); while (!observed[`TM_OUT_REQUEST_READY_OFFSET]);
    pins[`TM_IN_REQUEST_VALID_OFFSET+:`TM_IN_REQUEST_VALID_WIDTH] = 0;
    pins[`TM_IN_REQUEST_BITS_CMD_MEM_ADDR_OFFSET+:`TM_IN_REQUEST_BITS_CMD_MEM_ADDR_WIDTH] = '1;
  endtask
  task completed(int tag);
    do tick(); while (reference.pending.exists(tag));
    repeat (3) tick();
  endtask
  task body();
    pins = '0;
    pins[`TM_IN_RESPONSE_READY_OFFSET+:`TM_IN_RESPONSE_READY_WIDTH] = 1;
    repeat (3) tick(1);
    tick();
    for (int offset = 0; offset < 16; offset++) begin
      send(0, 0, 'h1000 + offset, 2, 0);
      completed(0);
      send(1, 1, 'h9000 + offset, 2, 1);
      completed(1);
      send(0, 0, 'h9000 + offset, 2, 0);
      completed(0);
    end
    send(0, 0, 'h1ff3, 3, 0, 64'd3 << 39);
    completed(0);
    send(1, 1, 'h8ff5, 3, 1, 64'd3 << 39);
    completed(1);
    send(0, 0, 'h3000, 3, 0, (64'd1 << 63) | (64'd1 << 58) | (64'd1 << 39));
    completed(0);
    send(1, 1, 'ha000, 3, 1, (64'd1 << 63) | (64'd1 << 58) | (64'd1 << 39));
    completed(1);
    send(0, 2, 'h4003, 2, 0,
         (64'd1 << 62) | (64'd2 << 59) | (64'd2 << 53) | (64'd7 << 43) | (64'd1 << 36));
    completed(0);
    for (int length = 4; length <= 68; length += 32) begin
      send(2, 3, 'h5000, length, 4);
      completed(2);
    end
    send(3, 4, 'h6000, 3, 0, (64'd7 << 56) | (64'd32 << 39));
    completed(3);
    send(2, 5, 0, 0, 4);
    completed(2);
    // Independent read and write producers retain their own ROB tags and data.
    send(0, 0, 'h7000, 4, 0);
    send(1, 1, 'hb000, 4, 1);
    completed(0);
    completed(1);
    pins[`TM_IN_RESPONSE_READY_OFFSET+:`TM_IN_RESPONSE_READY_WIDTH] = 0;
    send(0, 0, 'h7000, 1, 0);
    repeat (80) tick();
    pins[`TM_IN_RESPONSE_READY_OFFSET+:`TM_IN_RESPONSE_READY_WIDTH] = 1;
    completed(0);
    reference.image.read_error = 1;
    send(0, 0, 'h8000, 2, 0);
    completed(0);
    reference.image.read_error  = 0;
    reference.image.write_error = 1;
    send(1, 1, 'hc000, 2, 1);
    completed(1);
    reference.image.write_error = 0;
    reference.image.bank_error  = 1;
    send(0, 0, 'h8000, 2, 0);
    completed(0);
    reference.image.bank_error  = 0;
    reference.image.image_fault = 1;
    send(2, 3, 'h5000, 12, 5);
    completed(2);
    reference.image.image_fault = 0;
    reference.image.admission_error = 7;
    send(0, 0, 'h8000, 2, 0);
    completed(0);
    reference.image.admission_error  = 0;
    reference.image.completion_error = 10;
    send(0, 0, 'h8000, 1, 0);
    completed(0);
    reference.image.completion_error = 0;
    send(0, 0, 'h8000, 0, 0);
    completed(0);
    send(0, 0, 'h8000, 4, 0);
    repeat (8) tick();
    repeat (3) tick(1);
    tick();
    send(0, 0, 'h8000, 1, 0);
    completed(0);
    repeat (8) tick();
  endtask
endclass
