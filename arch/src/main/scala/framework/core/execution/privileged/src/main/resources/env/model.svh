class reference_model extends uvm_subscriber #(cycle_item);
  `uvm_component_utils(reference_model)
  chandle reference;
  int pmp_offsets[16][8] = `PRIV_PMP_OFFSETS;
  uvm_analysis_port #(result_item) expected_ap, actual_ap;
  // permitted answers the authorization of the previous cycle.
  bit [1:0] permitted;
  bit primed;
  covergroup behavior with function sample (
      int address, int operation, bit write, int privilege, bit trap, bit returning, bit interrupt
  );
    option.per_instance = 1;
    csr_operation: coverpoint operation {bins kinds[] = {[0 : 3]};}
    access_direction: coverpoint write;
    current_privilege: coverpoint privilege {
      bins user_mode = {0}; bins supervisor = {1}; bins machine_mode = {3};
    }
    trap_event: coverpoint trap;
    return_event: coverpoint returning;
    interrupt_event: coverpoint interrupt;
    pmp_address: coverpoint (address - 'h3b0) iff (address >= 'h3b0 && address <= 'h3bf && write) {
      bins entries[] = {[0 : 15]};
    }
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    expected_ap = new("expected_ap", this);
    actual_ap = new("actual_ap", this);
    behavior = new();
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    reference = privileged_create(`PRIV_ADDRESS_BITS);
  endfunction
  function void write(cycle_item t);
    result_item expected, actual;
    longint unsigned interrupts, cfg;
    if (t.reset) begin
      privileged_reset(reference);
      primed = 0;
      return;
    end
    expected = result_item::type_id::create("expected");
    actual = result_item::type_id::create("actual");
    expected.payload = '0;
    actual.payload = t.outputs;
    interrupts = (longint'(
    `PIN(t.inputs, INTERRUPTS_SOFTWARE)
    ) << 3) | (longint'(
    `PIN(t.inputs, INTERRUPTS_TIMER)
    ) << 7) | (longint'(
    `PIN(t.inputs, INTERRUPTS_EXTERNAL)
    ) << 11) | (longint'(
    `PIN(t.inputs, INTERRUPTS_SUPERVISOREXTERNAL)
    ) << 9);
    privileged_evaluate(
        reference, `PIN(t.inputs, CSR_ADDRESS), `PIN(t.inputs, CSR_OPERAND),
        `PIN(t.inputs, CSR_OPERATION), `PIN(t.inputs, CSR_WRITE), `PIN(t.inputs, CSR_COMMIT),
        `PIN(t.inputs, TRAP_VALID), `PIN(t.inputs, TRAP_BITS_PC), `PIN(t.inputs, TRAP_BITS_CAUSE),
        `PIN(t.inputs, TRAP_BITS_VALUE), `PIN(t.inputs, RETURNING_VALID),
        `PIN(t.inputs, RETURNING_BITS), `PIN(t.inputs, HARTID), interrupts, `PIN(t.inputs, RETIRED),
        `PIN(t.inputs, FLOATING_VALID), `PIN(t.inputs, FLOATING_BITS));
    expected.payload[`PRIV_OUT_CSR_DATA_OFFSET+:`PRIV_OUT_CSR_DATA_WIDTH] =
        privileged_value(reference, 0);
    expected.payload[`PRIV_OUT_CSR_ILLEGAL_OFFSET+:`PRIV_OUT_CSR_ILLEGAL_WIDTH] =
        privileged_value(reference, 1);
    expected.payload[`PRIV_OUT_CSR_MEMORYREAD_OFFSET +: `PRIV_OUT_CSR_MEMORYREAD_WIDTH] = privileged_value(
        reference, 2);
    expected.payload[`PRIV_OUT_REDIRECT_VALID_OFFSET +: `PRIV_OUT_REDIRECT_VALID_WIDTH] = privileged_value(
        reference, 3);
    expected.payload[`PRIV_OUT_REDIRECT_BITS_OFFSET +: `PRIV_OUT_REDIRECT_BITS_WIDTH] = privileged_value(
        reference, 4);
    expected.payload[`PRIV_OUT_INTERRUPT_VALID_OFFSET +: `PRIV_OUT_INTERRUPT_VALID_WIDTH] = privileged_value(
        reference, 5);
    expected.payload[`PRIV_OUT_INTERRUPT_BITS_OFFSET +: `PRIV_OUT_INTERRUPT_BITS_WIDTH] = privileged_value(
        reference, 6);
    expected.payload[`PRIV_OUT_WAKE_OFFSET+:`PRIV_OUT_WAKE_WIDTH] = privileged_value(reference, 7);
    expected.payload[`PRIV_OUT_PRIVILEGE_OFFSET+:`PRIV_OUT_PRIVILEGE_WIDTH] =
        privileged_value(reference, 8);
    expected.payload[`PRIV_OUT_DATAPRIVILEGE_OFFSET +: `PRIV_OUT_DATAPRIVILEGE_WIDTH] = privileged_value(
        reference, 9);
    expected.payload[`PRIV_OUT_SATP_OFFSET+:`PRIV_OUT_SATP_WIDTH] = privileged_value(reference, 10);
    expected.payload[`PRIV_OUT_SUM_OFFSET+:`PRIV_OUT_SUM_WIDTH] = privileged_value(reference, 11);
    expected.payload[`PRIV_OUT_MXR_OFFSET+:`PRIV_OUT_MXR_WIDTH] = privileged_value(reference, 12);
    expected.payload[`PRIV_OUT_FLOATINGENABLED_OFFSET +: `PRIV_OUT_FLOATINGENABLED_WIDTH] = privileged_value(
        reference, 13);
    expected.payload[`PRIV_OUT_ROUNDING_OFFSET+:`PRIV_OUT_ROUNDING_WIDTH] =
        privileged_value(reference, 14);
    expected.payload[`PRIV_OUT_TVM_OFFSET+:`PRIV_OUT_TVM_WIDTH] = privileged_value(reference, 15);
    expected.payload[`PRIV_OUT_TW_OFFSET+:`PRIV_OUT_TW_WIDTH] = privileged_value(reference, 16);
    expected.payload[`PRIV_OUT_TSR_OFFSET+:`PRIV_OUT_TSR_WIDTH] = privileged_value(reference, 17);
    expected.payload[`PRIV_OUT_PERMITTED_0_OFFSET+:`PRIV_OUT_PERMITTED_0_WIDTH] =
        primed ? permitted[0] : actual.payload[`PRIV_OUT_PERMITTED_0_OFFSET+:`PRIV_OUT_PERMITTED_0_WIDTH];
    expected.payload[`PRIV_OUT_PERMITTED_1_OFFSET+:`PRIV_OUT_PERMITTED_1_WIDTH] =
        primed ? permitted[1] : actual.payload[`PRIV_OUT_PERMITTED_1_OFFSET+:`PRIV_OUT_PERMITTED_1_WIDTH];
    permitted[0] = privileged_permitted(
        reference,
        `PIN(t.inputs, AUTHORIZATION_0_PADDR),
        `PIN(t.inputs, AUTHORIZATION_0_SIZE),
        `PIN(t.inputs, AUTHORIZATION_0_READ),
        `PIN(t.inputs, AUTHORIZATION_0_WRITE),
        `PIN(t.inputs, AUTHORIZATION_0_EXECUTE),
        `PIN(t.inputs, AUTHORIZATION_0_PRIVILEGE)
    );
    permitted[1] = privileged_permitted(
        reference,
        `PIN(t.inputs, AUTHORIZATION_1_PADDR),
        `PIN(t.inputs, AUTHORIZATION_1_SIZE),
        `PIN(t.inputs, AUTHORIZATION_1_READ),
        `PIN(t.inputs, AUTHORIZATION_1_WRITE),
        `PIN(t.inputs, AUTHORIZATION_1_EXECUTE),
        `PIN(t.inputs, AUTHORIZATION_1_PRIVILEGE)
    );
    primed = 1;
    for (int index = 0; index < 16; index++) begin
      cfg = privileged_pmp(reference, index, 0);
      expected.payload[pmp_offsets[index][0]] = cfg[7];
      expected.payload[pmp_offsets[index][1]+:2] = cfg[6:5];
      expected.payload[pmp_offsets[index][2]+:2] = cfg[4:3];
      expected.payload[pmp_offsets[index][3]] = cfg[2];
      expected.payload[pmp_offsets[index][4]] = cfg[1];
      expected.payload[pmp_offsets[index][5]] = cfg[0];
      expected.payload[pmp_offsets[index][6]+:(`PRIV_ADDRESS_BITS-2)] =
          privileged_pmp(reference, index, 1);
      expected.payload[pmp_offsets[index][7]+:`PRIV_ADDRESS_BITS] =
          privileged_pmp(reference, index, 2);
    end
    if (!actual.payload[`PRIV_OUT_REDIRECT_VALID_OFFSET+:`PRIV_OUT_REDIRECT_VALID_WIDTH]) begin
      actual.payload[`PRIV_OUT_REDIRECT_BITS_OFFSET+:`PRIV_OUT_REDIRECT_BITS_WIDTH] = 0;
    end
    if (!actual.payload[`PRIV_OUT_INTERRUPT_VALID_OFFSET+:`PRIV_OUT_INTERRUPT_VALID_WIDTH]) begin
      actual.payload[`PRIV_OUT_INTERRUPT_BITS_OFFSET+:`PRIV_OUT_INTERRUPT_BITS_WIDTH] = 0;
    end
    if (actual.payload[`PRIV_OUT_CSR_ILLEGAL_OFFSET+:`PRIV_OUT_CSR_ILLEGAL_WIDTH]) begin
      actual.payload[`PRIV_OUT_CSR_DATA_OFFSET+:`PRIV_OUT_CSR_DATA_WIDTH] = 0;
    end
    behavior.sample(
        `PIN(t.inputs, CSR_ADDRESS), `PIN(t.inputs, CSR_OPERATION), `PIN(t.inputs, CSR_WRITE),
        actual.payload[`PRIV_OUT_PRIVILEGE_OFFSET+:`PRIV_OUT_PRIVILEGE_WIDTH],
        `PIN(t.inputs, TRAP_VALID), `PIN(t.inputs, RETURNING_VALID),
        actual.payload[`PRIV_OUT_INTERRUPT_VALID_OFFSET+:`PRIV_OUT_INTERRUPT_VALID_WIDTH]);
    expected_ap.write(expected);
    actual_ap.write(actual);
    privileged_advance(reference);
  endfunction
  function void final_phase(uvm_phase phase);
    privileged_destroy(reference);
    super.final_phase(phase);
  endfunction
endclass
