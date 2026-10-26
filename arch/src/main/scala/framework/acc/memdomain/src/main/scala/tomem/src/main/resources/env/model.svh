class model extends uvm_subscriber #(cycle_item);
  `uvm_component_utils(model)
  fixture image;
  uvm_analysis_port #(result_item) expected, actual;
  uvm_analysis_port #(uvm_sequence_item) cancelled;
  typedef struct {
    int tag, bank_id, rows, groups, group_base, start_row, width, pixel, source_width;
    longint unsigned base, special;
    bit writing, is_2d, is_image, is_mmio, is_config;
    int words, index, fault;
    longint unsigned fault_address;
  } operation;
  operation load_op, store_op;
  bit load_active, store_active, pending[int];
  int
      load_serial,
      store_serial,
      read_beats,
      write_beats,
      image_words,
      bank_acks,
      bank_reads,
      metadata_acks;
  bit read_addresses[longint unsigned];
  longint unsigned write_address;
  byte unsigned memory[longint unsigned], store_bytes[longint unsigned];
  bit [127:0] banks[`TM_BANKS][`TM_GROUPS][`TM_ROWS];
  result_item load_results[$];
  covergroup behavior with function sample (
      int kind, int fault, bit overlap, bit stall, bit reset_busy
  );
    option.per_instance = 1;
    operations: coverpoint kind {bins all[] = {[0 : 5]};}
    faults: coverpoint fault {
      bins success = {0};
      bins access_fault = {2};
      bins bank_fault = {5};
      bins shape = {7};
      bins completion = {10};
    }
    concurrency: coverpoint overlap;
    backpressure: coverpoint stall;
    reset_cancellation: coverpoint reset_busy;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    expected = new("expected", this);
    actual = new("actual", this);
    cancelled = new("cancelled", this);
    behavior = new();
  endfunction
  function void start_of_simulation_phase(uvm_phase phase);
    foreach (banks[b, g, row]) banks[b][g][row] = image.initial_bank(b, g, row);
  endfunction
  function void verify(bit ok, string message);
    evidence::checks++;
    if (!ok) `uvm_fatal("TOMEM", message)
  endfunction
  function byte unsigned byte_at(longint unsigned address);
    return memory.exists(address) ? memory[address] : image.initial_byte(address);
  endfunction
  function bit [127:0] word_at(longint unsigned address);
    bit [127:0] value;
    for (int b = 0; b < 16; b++) value[b*8+:8] = byte_at(address + b);
    return value;
  endfunction
  function operation decode(cycle_item t);
    operation op;
    op.tag = `MI(t, REQUEST_BITS_ROB_ID);
    op.bank_id = `MI(t, REQUEST_BITS_CMD_BANK_ID);
    op.rows = `MI(t, REQUEST_BITS_CMD_ITER);
    op.base = `MI(t, REQUEST_BITS_CMD_MEM_ADDR);
    op.special = `MI(t, REQUEST_BITS_CMD_SPECIAL);
    op.writing = `MI(t, REQUEST_BITS_CMD_IS_STORE);
    op.is_2d = `MI(t, REQUEST_BITS_CMD_IS_MVIN_2D);
    op.is_image = `MI(t, REQUEST_BITS_CMD_IS_IMAGE);
    op.is_mmio = `MI(t, REQUEST_BITS_CMD_IS_MMIO);
    op.is_config = `MI(t, REQUEST_BITS_CMD_IS_CONFIG);
    op.width = ((op.special >> 59) & 7) + 1;
    op.pixel = ((op.special >> 36) & 127) * 8;
    op.source_width = (op.special >> 43) & 1023;
    op.group_base=op.special[63]&&!op.is_2d&&!op.is_image&&!op.is_mmio?int'((op.special>>58)&31):0;
    op.groups=op.is_image||op.is_mmio||op.is_config?1:(op.special[63]&&!op.is_2d?1:image.groups[op.bank_id]);
    op.start_row = op.is_2d ? int'((op.special >> 53) & 63) : 0;
    op.words = op.is_image ? (op.rows + 15) / 16 : op.rows * (op.is_2d ? op.width : op.groups);
    op.index = 0;
    op.fault = 0;
    op.fault_address = 0;
    if (!op.is_config) begin
      if (image.admission_error != 0) begin
        op.fault = image.admission_error;
        op.fault_address = 'hdead;
      end else if (op.rows == 0) begin
        op.fault = 7;
        op.fault_address = op.base;
      end else if (op.writing ? image.write_error : image.read_error) begin
        op.fault = 2;
        op.fault_address = ((op.base & ~64'd15) + image.physical_offset);
      end else if (!op.writing && image.bank_error) begin
        op.fault = 5;
        op.fault_address = op.base;
      end else if (op.is_image && image.image_fault) begin
        op.fault = 5;
        op.fault_address = 'hf00d;
      end
    end
    return op;
  endfunction
  function longint unsigned load_address(operation op, int index);
    if (op.is_image || op.is_mmio) return op.base + index * 16 + image.physical_offset;
    if (op.is_2d)
      return op.base+(index/op.width)*op.source_width*op.pixel+(index%op.width)*op.pixel+image.physical_offset;
    return op.base+(index/op.groups)*op.groups*((op.special>>39)&19'h7ffff)*16+(index%op.groups)*16+image.physical_offset;
  endfunction
  function void predict(cycle_item t);
    operation   op = decode(t);
    result_item result;
    verify(!pending.exists(op.tag), "Duplicate live ROB tag");
    pending[op.tag] = 1;
    if (op.writing) begin
      verify(!store_active, "Overlapping stores");
      store_op = op;
      store_active = 1;
      store_serial++;
      if (op.fault == 0 || image.write_error)
        for (int row = 0; row < op.rows; row++)
        for (int group_index = 0; group_index < op.groups; group_index++) begin
          bit [127:0] data = banks[op.bank_id][op.group_base+group_index][row];
          longint unsigned address=op.base+row*op.groups*((op.special>>39)&19'h7ffff)*16+group_index*16+image.physical_offset;
          for (int b = 0; b < 16; b++) store_bytes[address+b] = data[b*8+:8];
        end
    end else begin
      verify(!load_active, "Overlapping loads");
      load_op = op;
      load_active = 1;
      load_serial++;
      image_words = 0;
      read_addresses.delete();
      if (!op.is_config && image.admission_error == 0 && op.rows > 0)
        for (int i = 0; i < op.words; i++) begin
          longint unsigned address = load_address(op, i);
          for (int b = 0; b < 16; b++) read_addresses[(address+b)&~64'd15] = 1;
        end
      if(!op.is_config&&(op.fault==0||image.bank_error||(op.is_image&&image.image_fault)))begin
        int words = image.bank_error ? 1 : op.words;
        if (op.is_image) begin
          for (int i = 0; i < (op.rows + 3) / 4; i++) begin
            result = result_item::type_id::create("image_expected");
            result.value = 32'({
              byte_at(op.base + image.physical_offset + i * 4 + 3),
              byte_at(op.base + image.physical_offset + i * 4 + 2),
              byte_at(op.base + image.physical_offset + i * 4 + 1),
              byte_at(op.base + image.physical_offset + i * 4)
            });
            result.set_transaction_id((2 << 20) | (load_serial << 10) | i);
            expected.write(result);
          end
        end else
          for (int i = 0; i < words; i++) begin
            bit [127:0] data = word_at(load_address(op, i));
            int group_index = op.group_base + (op.is_2d ? 0 : i % op.groups);
            int row = op.start_row + (op.is_2d ? i : i / op.groups);
            if (op.is_2d && op.special[62]) data[127:64] = 0;
            result = result_item::type_id::create("load_expected");
            if (op.is_mmio)
              result.value = {
                64'(((op.special >> 39) & 17'h1ffff) + i * 16),
                data,
                16'((17'd1 << ((op.special >> 56) & 255)) - 1)
              };
            else result.value = {32'(op.bank_id), 32'(group_index), 32'(row), data, 16'hffff};
            result.set_transaction_id((1 << 20) | (load_serial << 10) | i);
            expected.write(result);
            load_results.push_back(result);
          end
      end
    end
    result = result_item::type_id::create("completion_expected");
    result.value = {
      32'(op.tag),
      32'(image.completion_error != 0 ? image.completion_error : op.fault),
      image.completion_error != 0 ? 64'hbeef : op.fault_address
    };
    result.set_transaction_id(op.tag);
    expected.write(result);
    behavior.sample(
        op.is_config ? 5 : op.is_image ? 3 : op.is_mmio ? 4 : op.is_2d ? 2 : op.writing ? 1 : 0,
        image.completion_error != 0 ? image.completion_error : op.fault,
        load_active && store_active, 0, 0);
  endfunction
  function void write(cycle_item t);
    if (t.reset) begin
      behavior.sample(0, 0, 0, 0, load_active || store_active);
      cancelled.write(null);
      load_active  = 0;
      store_active = 0;
      pending.delete();
      store_bytes.delete();
      load_results.delete();
      read_beats = 0;
      write_beats = 0;
      bank_acks = 0;
      bank_reads = 0;
      metadata_acks = 0;
      read_addresses.delete();
      return;
    end
    if (`MI(t, REQUEST_VALID) && `MO(t, REQUEST_READY)) predict(t);
    if (`MO(t, MEMORY_AXI_AR_VALID) && `MI(t, MEMORY_AXI_AR_READY)) begin
      verify(load_active, "AXI read without a load owner");
      verify(`MO(t, MEMORY_AXI_AR_BITS_SIZE) == 4 && `MO(t, MEMORY_AXI_AR_BITS_BURST) == 1 &&
             `MO(t, MEMORY_AXI_AR_BITS_ID) == 0, "AXI read attributes mismatch");
      verify(read_beats == 0, "Overlapping AXI read bursts");
      read_beats = `MO(t, MEMORY_AXI_AR_BITS_LEN) + 1;
      verify((`MO(t, MEMORY_AXI_AR_BITS_ADDR) & 4095) + read_beats * 16 <= 4096,
             "AXI read crossed a page");
      verify((`MO(t, MEMORY_AXI_AR_BITS_ADDR) & 15) == 0, "Unaligned AXI read");
      for (int i = 0; i < read_beats; i++)
      verify(read_addresses.exists(`MO(t, MEMORY_AXI_AR_BITS_ADDR) + i * 16),
             "AXI read escaped the accepted command footprint");
    end
    if (`MI(t, MEMORY_AXI_R_VALID) && `MO(t, MEMORY_AXI_R_READY)) begin
      verify(read_beats > 0, "AXI response without a request");
      verify(`MI(t, MEMORY_AXI_R_BITS_LAST) == (read_beats == 1), "RLAST mismatch");
      read_beats--;
    end
    if (`MO(t, MEMORY_AXI_AW_VALID) && `MI(t, MEMORY_AXI_AW_READY)) begin
      verify(store_active && write_beats == 0, "AXI write without a store owner");
      write_address = `MO(t, MEMORY_AXI_AW_BITS_ADDR);
      write_beats   = `MO(t, MEMORY_AXI_AW_BITS_LEN) + 1;
      verify(`MO(t, MEMORY_AXI_AW_BITS_SIZE) == 4 && `MO(t, MEMORY_AXI_AW_BITS_BURST) == 1 &&
             `MO(t, MEMORY_AXI_AW_BITS_ID) == 1, "AXI write attributes mismatch");
      verify((write_address & 4095) + write_beats * 16 <= 4096, "AXI write crossed a page");
    end
    if (`MO(t, MEMORY_AXI_W_VALID) && `MI(t, MEMORY_AXI_W_READY)) begin
      verify(write_beats > 0, "Write data without AW");
      verify(`MO(t, MEMORY_AXI_W_BITS_LAST) == (write_beats == 1), "WLAST mismatch");
      for (int b = 0; b < 16; b++) begin
        bit expected_strobe = store_bytes.exists(write_address + b);
        verify(t.outputs[`TM_OUT_MEMORY_AXI_W_BITS_STRB_OFFSET+b] == expected_strobe,
               "Store byte strobe mismatch");
        if (expected_strobe) begin
          verify(
              t.outputs[`TM_OUT_MEMORY_AXI_W_BITS_DATA_OFFSET+b*8+:8]===store_bytes[write_address+b],
              "Store data differs from logical bank contents");
          memory[write_address+b] = store_bytes[write_address+b];
          store_bytes.delete(write_address + b);
        end
      end
      write_address += 16;
      write_beats--;
    end
    if (`MO(t, BANKREAD_READ_REQ_VALID) && `MI(t, BANKREAD_READ_REQ_READY)) begin
      int i = store_op.index;
      verify(store_active, "Bank read without a store owner");
      verify(`MO(t, BANKREAD_BANK_ID) == store_op.bank_id && `MO(t, BANKREAD_GROUP_ID)
             == store_op.group_base + i % store_op.groups && `MO(t, BANKREAD_READ_REQ_BITS_ADDR)
             == i / store_op.groups, "Store bank address/group mismatch");
      store_op.index++;
      bank_reads++;
    end
    if (`MO(t, BANKWRITE_WRITE_REQ_VALID) && `MI(t, BANKWRITE_WRITE_REQ_READY)) begin
      result_item result = result_item::type_id::create("load_actual");
      result_item plan;
      verify(load_active && load_results.size() != 0, "Unexpected bank write");
      plan = load_results.pop_front();
      result.value = {
        32'(`MO(t, BANKWRITE_BANK_ID)),
        32'(`MO(t, BANKWRITE_GROUP_ID)),
        32'(`MO(t, BANKWRITE_WRITE_REQ_BITS_ADDR)),
        128'(`MO(t, BANKWRITE_WRITE_REQ_BITS_DATA)),
        16'hffff
      };
      for (int b = 0; b < 16; b++)
      verify(t.outputs[`TM_OUT_BANKWRITE_WRITE_REQ_BITS_MASK_0_OFFSET+b],
             "Load did not replace the whole bank row");
      result.set_transaction_id(plan.get_transaction_id());
      actual.write(result);
      if (!image.bank_error)
        banks[
        `MO(t, BANKWRITE_BANK_ID)
        ][
        `MO(t, BANKWRITE_GROUP_ID)
        ][
        `MO(t, BANKWRITE_WRITE_REQ_BITS_ADDR)
        ] =
        `MO(t, BANKWRITE_WRITE_REQ_BITS_DATA);
      load_op.index++;
      bank_acks++;
    end
    if (`MO(t, METADATA_REQUEST_VALID) && `MI(t, METADATA_REQUEST_READY)) begin
      result_item result = result_item::type_id::create("metadata_actual");
      result_item plan;
      verify(load_active && load_op.is_mmio && load_results.size() != 0,
             "Unexpected metadata write");
      plan = load_results.pop_front();
      result.value = {
        64'(`MO(t, METADATA_REQUEST_BITS_ADDR)),
        128'(`MO(t, METADATA_REQUEST_BITS_DATA)),
        16'(`MO(t, METADATA_REQUEST_BITS_MASK))
      };
      result.set_transaction_id(plan.get_transaction_id());
      actual.write(result);
      load_op.index++;
      metadata_acks++;
    end
    if (`MI(t, BANKREAD_READ_RESP_VALID) && `MO(t, BANKREAD_READ_RESP_READY)) begin
      verify(bank_reads > 0, "Unexpected bank read response");
      bank_reads--;
    end
    if (`MI(t, BANKWRITE_WRITE_RESP_VALID) && `MO(t, BANKWRITE_WRITE_RESP_READY)) begin
      verify(bank_acks > 0, "Unexpected bank write acknowledgment");
      bank_acks--;
    end
    if (`MI(t, METADATA_RESPONSE_VALID) && `MO(t, METADATA_RESPONSE_READY)) begin
      verify(metadata_acks > 0, "Unexpected metadata acknowledgment");
      metadata_acks--;
    end
    if (`MO(t, IMAGE_START_VALID) && `MI(t, IMAGE_START_READY))
      verify(load_active && load_op.is_image && `MO(t, IMAGE_START_BITS_BYTES) == load_op.rows &&
             `MO(t, IMAGE_START_BITS_BUFFER) == load_op.bank_id - 4,
             "Image start descriptor mismatch");
    if (`MO(t, IMAGE_TERMINAL_VALID) && `MI(t, IMAGE_TERMINAL_READY)) begin
      int terminal_error = image.image_fault && load_op.fault == 5 ? 0 : load_op.fault;
      verify(`MO(t, IMAGE_TERMINAL_BITS_ERROR) == terminal_error &&
             `MO(t, IMAGE_TERMINAL_BITS_ADDRESS)
             == (terminal_error == 0 ? 64'd0 : load_op.fault_address),
             "Image terminal status mismatch");
    end
    if (`MO(t, IMAGE_DATA_VALID) && `MI(t, IMAGE_DATA_READY)) begin
      result_item result = result_item::type_id::create("image_actual");
      verify(load_active && load_op.is_image, "Image data without an owner");
      result.value = `MO(t, IMAGE_DATA_BITS);
      result.set_transaction_id((2 << 20) | (load_serial << 10) | image_words);
      actual.write(result);
      image_words++;
    end
    if (`MO(t, MEMORY_CONTROL_FINISHED_0_VALID)) begin
      operation op = load_op;
      verify(load_active, "Completion without an owner");
      verify(`MO(t, MEMORY_CONTROL_FINISHED_0_BITS_ERROR) == op.fault &&
             `MO(t, MEMORY_CONTROL_FINISHED_0_BITS_ADDRESS) == op.fault_address,
             "Finished status differs from DMA/bank result");
      verify(load_results.size() == 0 && read_beats == 0 && bank_acks == 0 && metadata_acks == 0,
             "Finished before data drained");
    end
    if (`MO(t, MEMORY_CONTROL_FINISHED_1_VALID)) begin
      operation op = store_op;
      verify(store_active, "Completion without an owner");
      verify(`MO(t, MEMORY_CONTROL_FINISHED_1_BITS_ERROR) == op.fault &&
             `MO(t, MEMORY_CONTROL_FINISHED_1_BITS_ADDRESS) == op.fault_address,
             "Finished status differs from DMA/bank result");
      verify(store_bytes.num() == 0 && write_beats == 0 && bank_reads == 0,
             "Finished before data drained");
    end
    if (`MO(t, MEMORY_CONTROL_FINISHED_2_VALID)) begin
      operation op = load_op;
      verify(load_active, "Completion without an owner");
      verify(`MO(t, MEMORY_CONTROL_FINISHED_2_BITS_ERROR) == op.fault &&
             `MO(t, MEMORY_CONTROL_FINISHED_2_BITS_ADDRESS) == op.fault_address,
             "Finished status differs from DMA/bank result");
      verify(load_results.size() == 0 && read_beats == 0 && bank_acks == 0 && metadata_acks == 0,
             "Finished before data drained");
    end
    if (`MO(t, RESPONSE_VALID) && `MI(t, RESPONSE_READY)) begin
      result_item result = result_item::type_id::create("completion_actual");
      int tag = `MO(t, RESPONSE_BITS_ROB_ID);
      verify(pending.exists(tag), "Unsolicited command completion");
      result.value = {
        32'(tag), 32'(`MO(t, RESPONSE_BITS_FAULT_ERROR)), 64'(`MO(t, RESPONSE_BITS_FAULT_ADDRESS))
      };
      result.set_transaction_id(tag);
      actual.write(result);
      pending.delete(tag);
      if (store_active && store_op.tag == tag) store_active = 0;
      else begin
        verify(load_active && load_op.tag == tag, "Completion owner mismatch");
        load_active = 0;
      end
    end
    if (`MO(t, RESPONSE_VALID) && !`MI(t, RESPONSE_READY))
      behavior.sample(0, 0, load_active && store_active, 1, 0);
  endfunction
  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    verify(
        pending.num()==0&&!load_active&&!store_active&&load_results.size()==0&&store_bytes.num()==0,
        "ToMem did not drain");
  endfunction
endclass
