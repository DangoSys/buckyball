class peer extends uvm_component;
  `uvm_component_utils(peer)
  virtual tomem_if vif;
  fixture image;
  bit [127:0] bank_memory[`TM_BANKS][`TM_GROUPS][`TM_ROWS];
  byte unsigned memory[longint unsigned];
  int cycle, read_beats, write_beats, rid, bid, bank_due, bank_ack_due, metadata_due, image_due;
  longint unsigned read_address, write_address;
  bit b_pending, bank_pending, bank_ack, metadata_pending, image_pending;
  bit [127:0] bank_data;
  bit complete_pending[3], complete_armed[3];
  int complete_due[3], complete_error[3];
  longint unsigned complete_address[3];
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual tomem_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "ToMem interface is required")
  endfunction
  function void start_of_simulation_phase(uvm_phase phase);
    foreach (bank_memory[b, g, row]) bank_memory[b][g][row] = image.initial_bank(b, g, row);
  endfunction
  function bit [127:0] word_at(longint unsigned address);
    bit [127:0] value;
    for (int b = 0; b < 16; b++)
    value[b*8+:8] = memory.exists(address + b) ? memory[address+b] :
        image.initial_byte(address + b);
    return value;
  endfunction
  task run_phase(uvm_phase phase);
    forever begin
      @(posedge vif.clock);
      cycle++;
      if (vif.reset) begin
        read_beats = 0;
        write_beats = 0;
        b_pending = 0;
        bank_pending = 0;
        bank_ack = 0;
        metadata_pending = 0;
        image_pending = 0;
        foreach (complete_pending[p]) begin
          complete_pending[p] = 0;
          complete_armed[p]   = 1;
        end
      end else begin
        if (`TO(MEMORY_AXI_R_READY) && `TI(MEMORY_AXI_R_VALID)) begin
          read_beats--;
          read_address += 16;
        end
        if (`TO(MEMORY_AXI_B_READY) && `TI(MEMORY_AXI_B_VALID)) b_pending = 0;
        if (`TO(MEMORY_AXI_AR_VALID) && `TI(MEMORY_AXI_AR_READY)) begin
          read_address = `TO(MEMORY_AXI_AR_BITS_ADDR);
          read_beats = `TO(MEMORY_AXI_AR_BITS_LEN) + 1;
          rid = `TO(MEMORY_AXI_AR_BITS_ID);
        end
        if (`TO(MEMORY_AXI_AW_VALID) && `TI(MEMORY_AXI_AW_READY)) begin
          write_address = `TO(MEMORY_AXI_AW_BITS_ADDR);
          write_beats = `TO(MEMORY_AXI_AW_BITS_LEN) + 1;
          bid = `TO(MEMORY_AXI_AW_BITS_ID);
        end
        if (`TO(MEMORY_AXI_W_VALID) && `TI(MEMORY_AXI_W_READY)) begin
          for (int b = 0; b < 16; b++)
          if (vif.outputs[`TM_OUT_MEMORY_AXI_W_BITS_STRB_OFFSET+b])
            memory[write_address+b] = vif.outputs[`TM_OUT_MEMORY_AXI_W_BITS_DATA_OFFSET+b*8+:8];
          write_address += 16;
          write_beats--;
          if (write_beats == 0) b_pending = 1;
        end
        if (`TO(BANKREAD_READ_RESP_READY) && `TI(BANKREAD_READ_RESP_VALID)) bank_pending = 0;
        if (`TO(BANKREAD_READ_REQ_VALID) && `TI(BANKREAD_READ_REQ_READY)) begin
          bank_data = bank_memory[
          `TO(BANKREAD_BANK_ID)
          ][
          `TO(BANKREAD_GROUP_ID)
          ][
          `TO(BANKREAD_READ_REQ_BITS_ADDR)
          ];
          bank_pending = 1;
          bank_due = cycle + 2;
        end
        if (`TO(BANKWRITE_WRITE_RESP_READY) && `TI(BANKWRITE_WRITE_RESP_VALID)) bank_ack = 0;
        if (`TO(BANKWRITE_WRITE_REQ_VALID) && `TI(BANKWRITE_WRITE_REQ_READY)) begin
          if (!image.bank_error)
            bank_memory[
            `TO(BANKWRITE_BANK_ID)
            ][
            `TO(BANKWRITE_GROUP_ID)
            ][
            `TO(BANKWRITE_WRITE_REQ_BITS_ADDR)
            ] =
            `TO(BANKWRITE_WRITE_REQ_BITS_DATA);
          bank_ack = 1;
          bank_ack_due = cycle + 3;
        end
        if (`TO(METADATA_RESPONSE_READY) && `TI(METADATA_RESPONSE_VALID)) metadata_pending = 0;
        if (`TO(METADATA_REQUEST_VALID) && `TI(METADATA_REQUEST_READY)) begin
          metadata_pending = 1;
          metadata_due = cycle + 2;
        end
        if (`TO(IMAGE_COMPLETE_READY) && `TI(IMAGE_COMPLETE_VALID)) image_pending = 0;
        if (`TO(IMAGE_TERMINAL_VALID) && `TI(IMAGE_TERMINAL_READY)) begin
          image_pending = 1;
          image_due = cycle + 3;
        end
        if (`TO(MEMORY_CONTROL_COMPLETE_0_READY) && `TI(MEMORY_CONTROL_COMPLETE_0_VALID))
          complete_pending[0] = 0;
        if (!`TO(MEMORY_CONTROL_FINISHED_0_VALID)) complete_armed[0] = 1;
        if (`TO(MEMORY_CONTROL_FINISHED_0_VALID) && complete_armed[0] && !complete_pending[0]) begin
          complete_armed[0] = 0;
          complete_pending[0] = 1;
          complete_due[0] = cycle + 6;
          complete_error[0] = image.completion_error != 0 ? image.completion_error : `TO(
              MEMORY_CONTROL_FINISHED_0_BITS_ERROR);
          complete_address[0] = image.completion_error != 0 ? 64'hbeef : `TO(
              MEMORY_CONTROL_FINISHED_0_BITS_ADDRESS);
        end
        if (`TO(MEMORY_CONTROL_COMPLETE_1_READY) && `TI(MEMORY_CONTROL_COMPLETE_1_VALID))
          complete_pending[1] = 0;
        if (!`TO(MEMORY_CONTROL_FINISHED_1_VALID)) complete_armed[1] = 1;
        if (`TO(MEMORY_CONTROL_FINISHED_1_VALID) && complete_armed[1] && !complete_pending[1]) begin
          complete_armed[1] = 0;
          complete_pending[1] = 1;
          complete_due[1] = cycle + 6;
          complete_error[1] = image.completion_error != 0 ? image.completion_error : `TO(
              MEMORY_CONTROL_FINISHED_1_BITS_ERROR);
          complete_address[1] = image.completion_error != 0 ? 64'hbeef : `TO(
              MEMORY_CONTROL_FINISHED_1_BITS_ADDRESS);
        end
        if (`TO(MEMORY_CONTROL_COMPLETE_2_READY) && `TI(MEMORY_CONTROL_COMPLETE_2_VALID))
          complete_pending[2] = 0;
        if (!`TO(MEMORY_CONTROL_FINISHED_2_VALID)) complete_armed[2] = 1;
        if (`TO(MEMORY_CONTROL_FINISHED_2_VALID) && complete_armed[2] && !complete_pending[2]) begin
          complete_armed[2] = 0;
          complete_pending[2] = 1;
          complete_due[2] = cycle + 6;
          complete_error[2] = image.completion_error != 0 ? image.completion_error : `TO(
              MEMORY_CONTROL_FINISHED_2_BITS_ERROR);
          complete_address[2] = image.completion_error != 0 ? 64'hbeef : `TO(
              MEMORY_CONTROL_FINISHED_2_BITS_ADDRESS);
        end
      end
      @(negedge vif.clock);
      vif.backend = '0;
      vif.backend[`TM_IN_QUERYGROUPS_OFFSET+:`TM_IN_QUERYGROUPS_WIDTH] = image.groups[
      `TO(QUERYVBANK)
      ];
      vif.backend[`TM_IN_MEMORY_AXI_AR_READY_OFFSET+:`TM_IN_MEMORY_AXI_AR_READY_WIDTH] = read_beats == 0 && cycle % 4 != 0;
      vif.backend[`TM_IN_MEMORY_AXI_R_VALID_OFFSET+:`TM_IN_MEMORY_AXI_R_VALID_WIDTH] = read_beats > 0;
      vif.backend[`TM_IN_MEMORY_AXI_R_BITS_DATA_OFFSET+:`TM_IN_MEMORY_AXI_R_BITS_DATA_WIDTH] = word_at(
          read_address);
      vif.backend[`TM_IN_MEMORY_AXI_R_BITS_ID_OFFSET+:`TM_IN_MEMORY_AXI_R_BITS_ID_WIDTH] = rid;
      vif.backend[`TM_IN_MEMORY_AXI_R_BITS_RESP_OFFSET+:`TM_IN_MEMORY_AXI_R_BITS_RESP_WIDTH] = image.read_error ? 2 : 0;
      vif.backend[`TM_IN_MEMORY_AXI_R_BITS_LAST_OFFSET+:`TM_IN_MEMORY_AXI_R_BITS_LAST_WIDTH] = read_beats == 1;
      vif.backend[`TM_IN_MEMORY_AXI_AW_READY_OFFSET+:`TM_IN_MEMORY_AXI_AW_READY_WIDTH] = write_beats == 0 && !b_pending && cycle % 5 != 0;
      vif.backend[`TM_IN_MEMORY_AXI_W_READY_OFFSET+:`TM_IN_MEMORY_AXI_W_READY_WIDTH] = write_beats > 0 && cycle % 3 != 0;
      vif.backend[`TM_IN_MEMORY_AXI_B_VALID_OFFSET+:`TM_IN_MEMORY_AXI_B_VALID_WIDTH] = b_pending && cycle % 4 != 0;
      vif.backend[`TM_IN_MEMORY_AXI_B_BITS_ID_OFFSET+:`TM_IN_MEMORY_AXI_B_BITS_ID_WIDTH] = bid;
      vif.backend[`TM_IN_MEMORY_AXI_B_BITS_RESP_OFFSET+:`TM_IN_MEMORY_AXI_B_BITS_RESP_WIDTH] = image.write_error ? 2 : 0;
      vif.backend[`TM_IN_BANKREAD_READ_REQ_READY_OFFSET+:`TM_IN_BANKREAD_READ_REQ_READY_WIDTH] = !bank_pending && cycle % 3 != 0;
      vif.backend[`TM_IN_BANKREAD_READ_RESP_VALID_OFFSET+:`TM_IN_BANKREAD_READ_RESP_VALID_WIDTH] = bank_pending && cycle >= bank_due;
      vif.backend[`TM_IN_BANKREAD_READ_RESP_BITS_DATA_OFFSET+:`TM_IN_BANKREAD_READ_RESP_BITS_DATA_WIDTH] = bank_data;
      vif.backend[`TM_IN_BANKWRITE_WRITE_REQ_READY_OFFSET+:`TM_IN_BANKWRITE_WRITE_REQ_READY_WIDTH] = !bank_ack && cycle % 4 != 0;
      vif.backend[`TM_IN_BANKWRITE_WRITE_RESP_VALID_OFFSET+:`TM_IN_BANKWRITE_WRITE_RESP_VALID_WIDTH] = bank_ack && cycle >= bank_ack_due;
      vif.backend[`TM_IN_BANKWRITE_WRITE_RESP_BITS_OK_OFFSET+:`TM_IN_BANKWRITE_WRITE_RESP_BITS_OK_WIDTH] = !image.bank_error;
      vif.backend[`TM_IN_METADATA_REQUEST_READY_OFFSET+:`TM_IN_METADATA_REQUEST_READY_WIDTH] = !metadata_pending && cycle % 3 != 0;
      vif.backend[`TM_IN_METADATA_RESPONSE_VALID_OFFSET+:`TM_IN_METADATA_RESPONSE_VALID_WIDTH] = metadata_pending && cycle >= metadata_due;
      vif.backend[`TM_IN_METADATA_RESPONSE_BITS_OFFSET+:`TM_IN_METADATA_RESPONSE_BITS_WIDTH] = !image.bank_error;
      vif.backend[`TM_IN_IMAGE_START_READY_OFFSET+:`TM_IN_IMAGE_START_READY_WIDTH] = cycle % 4 != 0;
      vif.backend[`TM_IN_IMAGE_DATA_READY_OFFSET+:`TM_IN_IMAGE_DATA_READY_WIDTH] = cycle % 5 != 0;
      vif.backend[`TM_IN_IMAGE_TERMINAL_READY_OFFSET+:`TM_IN_IMAGE_TERMINAL_READY_WIDTH] = cycle % 3 != 0;
      vif.backend[`TM_IN_IMAGE_COMPLETE_VALID_OFFSET+:`TM_IN_IMAGE_COMPLETE_VALID_WIDTH] = image_pending && cycle >= image_due;
      vif.backend[`TM_IN_IMAGE_COMPLETE_BITS_FAULT_OFFSET+:`TM_IN_IMAGE_COMPLETE_BITS_FAULT_WIDTH] = image.image_fault;
      vif.backend[`TM_IN_IMAGE_COMPLETE_BITS_CAUSE_OFFSET+:`TM_IN_IMAGE_COMPLETE_BITS_CAUSE_WIDTH] = 5;
      vif.backend[`TM_IN_IMAGE_COMPLETE_BITS_TVAL_OFFSET+:`TM_IN_IMAGE_COMPLETE_BITS_TVAL_WIDTH] = 'hf00d;
      vif.backend[`TM_IN_MEMORY_CONTROL_DECISIONS_0_VALID_OFFSET+:`TM_IN_MEMORY_CONTROL_DECISIONS_0_VALID_WIDTH] = 1;
      vif.backend[`TM_IN_MEMORY_CONTROL_DECISIONS_0_PARENTID_OFFSET+:`TM_IN_MEMORY_CONTROL_DECISIONS_0_PARENTID_WIDTH] = 0;
      vif.backend[`TM_IN_MEMORY_CONTROL_DECISIONS_0_FAULT_ERROR_OFFSET+:`TM_IN_MEMORY_CONTROL_DECISIONS_0_FAULT_ERROR_WIDTH] = image.admission_error;
      vif.backend[`TM_IN_MEMORY_CONTROL_DECISIONS_0_FAULT_ADDRESS_OFFSET+:`TM_IN_MEMORY_CONTROL_DECISIONS_0_FAULT_ADDRESS_WIDTH] = image.admission_error == 0 ? 0 : 64'hdead;
      vif.backend[`TM_IN_MEMORY_CONTROL_COMPLETE_0_VALID_OFFSET+:`TM_IN_MEMORY_CONTROL_COMPLETE_0_VALID_WIDTH] = complete_pending[0] && cycle >= complete_due[0];
      vif.backend[`TM_IN_MEMORY_CONTROL_COMPLETE_0_BITS_ERROR_OFFSET+:`TM_IN_MEMORY_CONTROL_COMPLETE_0_BITS_ERROR_WIDTH] = complete_error[0];
      vif.backend[`TM_IN_MEMORY_CONTROL_COMPLETE_0_BITS_ADDRESS_OFFSET+:`TM_IN_MEMORY_CONTROL_COMPLETE_0_BITS_ADDRESS_WIDTH] = complete_address[0];
      vif.backend[`TM_IN_MEMORY_CONTROL_DECISIONS_1_VALID_OFFSET+:`TM_IN_MEMORY_CONTROL_DECISIONS_1_VALID_WIDTH] = 1;
      vif.backend[`TM_IN_MEMORY_CONTROL_DECISIONS_1_PARENTID_OFFSET+:`TM_IN_MEMORY_CONTROL_DECISIONS_1_PARENTID_WIDTH] = 1;
      vif.backend[`TM_IN_MEMORY_CONTROL_DECISIONS_1_FAULT_ERROR_OFFSET+:`TM_IN_MEMORY_CONTROL_DECISIONS_1_FAULT_ERROR_WIDTH] = image.admission_error;
      vif.backend[`TM_IN_MEMORY_CONTROL_DECISIONS_1_FAULT_ADDRESS_OFFSET+:`TM_IN_MEMORY_CONTROL_DECISIONS_1_FAULT_ADDRESS_WIDTH] = image.admission_error == 0 ? 0 : 64'hdead;
      vif.backend[`TM_IN_MEMORY_CONTROL_COMPLETE_1_VALID_OFFSET+:`TM_IN_MEMORY_CONTROL_COMPLETE_1_VALID_WIDTH] = complete_pending[1] && cycle >= complete_due[1];
      vif.backend[`TM_IN_MEMORY_CONTROL_COMPLETE_1_BITS_ERROR_OFFSET+:`TM_IN_MEMORY_CONTROL_COMPLETE_1_BITS_ERROR_WIDTH] = complete_error[1];
      vif.backend[`TM_IN_MEMORY_CONTROL_COMPLETE_1_BITS_ADDRESS_OFFSET+:`TM_IN_MEMORY_CONTROL_COMPLETE_1_BITS_ADDRESS_WIDTH] = complete_address[1];
      vif.backend[`TM_IN_MEMORY_CONTROL_DECISIONS_2_VALID_OFFSET+:`TM_IN_MEMORY_CONTROL_DECISIONS_2_VALID_WIDTH] = 1;
      vif.backend[`TM_IN_MEMORY_CONTROL_DECISIONS_2_PARENTID_OFFSET+:`TM_IN_MEMORY_CONTROL_DECISIONS_2_PARENTID_WIDTH] = 2;
      vif.backend[`TM_IN_MEMORY_CONTROL_DECISIONS_2_FAULT_ERROR_OFFSET+:`TM_IN_MEMORY_CONTROL_DECISIONS_2_FAULT_ERROR_WIDTH] = image.admission_error;
      vif.backend[`TM_IN_MEMORY_CONTROL_DECISIONS_2_FAULT_ADDRESS_OFFSET+:`TM_IN_MEMORY_CONTROL_DECISIONS_2_FAULT_ADDRESS_WIDTH] = image.admission_error == 0 ? 0 : 64'hdead;
      vif.backend[`TM_IN_MEMORY_CONTROL_COMPLETE_2_VALID_OFFSET+:`TM_IN_MEMORY_CONTROL_COMPLETE_2_VALID_WIDTH] = complete_pending[2] && cycle >= complete_due[2];
      vif.backend[`TM_IN_MEMORY_CONTROL_COMPLETE_2_BITS_ERROR_OFFSET+:`TM_IN_MEMORY_CONTROL_COMPLETE_2_BITS_ERROR_WIDTH] = complete_error[2];
      vif.backend[`TM_IN_MEMORY_CONTROL_COMPLETE_2_BITS_ADDRESS_OFFSET+:`TM_IN_MEMORY_CONTROL_COMPLETE_2_BITS_ADDRESS_WIDTH] = complete_address[2];
      vif.backend[`TM_IN_MEMORY_CONTROL_MAPPINGS_0_HIT_OFFSET+:`TM_IN_MEMORY_CONTROL_MAPPINGS_0_HIT_WIDTH] = 1;
      vif.backend[`TM_IN_MEMORY_CONTROL_MAPPINGS_0_PA_OFFSET+:`TM_IN_MEMORY_CONTROL_MAPPINGS_0_PA_WIDTH] =
      `TO(MEMORY_CONTROL_QUERIES_0_VA)
      + image.physical_offset;
      vif.backend[`TM_IN_MEMORY_CONTROL_MAPPINGS_0_ERROR_OFFSET+:`TM_IN_MEMORY_CONTROL_MAPPINGS_0_ERROR_WIDTH] = 0;
      vif.backend[`TM_IN_MEMORY_CONTROL_MAPPINGS_1_HIT_OFFSET+:`TM_IN_MEMORY_CONTROL_MAPPINGS_1_HIT_WIDTH] = 1;
      vif.backend[`TM_IN_MEMORY_CONTROL_MAPPINGS_1_PA_OFFSET+:`TM_IN_MEMORY_CONTROL_MAPPINGS_1_PA_WIDTH] =
      `TO(MEMORY_CONTROL_QUERIES_1_VA)
      + image.physical_offset;
      vif.backend[`TM_IN_MEMORY_CONTROL_MAPPINGS_1_ERROR_OFFSET+:`TM_IN_MEMORY_CONTROL_MAPPINGS_1_ERROR_WIDTH] = 0;
    end
  endtask
endclass
