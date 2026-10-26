class model extends uvm_subscriber #(cycle_item);
  `uvm_component_utils(model)
  uvm_analysis_port #(result_item) expected, actual;
  const int copies[`AD_PAYLOAD_FIELDS][4] = `AD_PAYLOAD_MAP;
  bit pending, moving, has_result, work_done, response_done, move_error, remote_owner;
  int saved_tag, destination, tags[$];
  covergroup behavior with function sample (int route, bit stalled, bit cancelled);
    option.per_instance = 1;
    routes: coverpoint route {bins all[] = {[0 : 0]};}
    backpressure: coverpoint stalled;
    reset_cancellation: coverpoint cancelled;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    expected = new("expected", this);
    actual   = new("actual", this);
    behavior = new();
  endfunction
  function void write(cycle_item t);
    logic [`AD_OUTPUT_BITS-1:0] want, mask;
    result_item predicted, received;

    if (t.reset) begin
      behavior.sample(0, 0, pending || tags.size() != 0);
      pending = 0;
      tags.delete();
      return;
    end
    want = '0;
    mask = '0;
    begin
      want[`AD_OUT_MOVE_COMMAND_VALID_OFFSET+:`AD_OUT_MOVE_COMMAND_VALID_WIDTH] =
      `AI(t, ROCC_CMD_VALID)
      && !pending;
      mask[`AD_OUT_MOVE_COMMAND_VALID_OFFSET+:`AD_OUT_MOVE_COMMAND_VALID_WIDTH] = '1;

      want[`AD_OUT_ROCC_CMD_READY_OFFSET+:`AD_OUT_ROCC_CMD_READY_WIDTH] =
      `AI(t, MOVE_COMMAND_READY)
      && !pending;
      mask[`AD_OUT_ROCC_CMD_READY_OFFSET+:`AD_OUT_ROCC_CMD_READY_WIDTH] = '1;

      want[`AD_OUT_MOVE_COMMAND_BITS_ROWS_OFFSET+:`AD_OUT_MOVE_COMMAND_BITS_ROWS_WIDTH]=
              {1'b0, t.inputs[`AD_IN_ROCC_CMD_BITS_INSTRUCTION_RS2DATA_OFFSET+32+:16]} + 17'd1;
      mask[`AD_OUT_MOVE_COMMAND_BITS_ROWS_OFFSET+:`AD_OUT_MOVE_COMMAND_BITS_ROWS_WIDTH] = '1;

      want[`AD_OUT_MOVE_COMPLETION_READY_OFFSET+:`AD_OUT_MOVE_COMPLETION_READY_WIDTH] = pending;
      mask[`AD_OUT_MOVE_COMPLETION_READY_OFFSET+:`AD_OUT_MOVE_COMPLETION_READY_WIDTH] = '1;

      want[`AD_OUT_ROCC_BUSY_OFFSET+:`AD_OUT_ROCC_BUSY_WIDTH] = pending;
      mask[`AD_OUT_ROCC_BUSY_OFFSET+:`AD_OUT_ROCC_BUSY_WIDTH] = '1;

      want[`AD_OUT_ROCC_INTERRUPT_OFFSET+:`AD_OUT_ROCC_INTERRUPT_WIDTH] = 0;
      mask[`AD_OUT_ROCC_INTERRUPT_OFFSET+:`AD_OUT_ROCC_INTERRUPT_WIDTH] = '1;

      want[`AD_OUT_ROCC_RESP_VALID_OFFSET+:`AD_OUT_ROCC_RESP_VALID_WIDTH] = 0;
      mask[`AD_OUT_ROCC_RESP_VALID_OFFSET+:`AD_OUT_ROCC_RESP_VALID_WIDTH] = '1;


    end
    foreach (copies[field])
    for (int b = 0; b < copies[field][2]; b++)
    if (want[copies[field][3]]) begin
      want[copies[field][1]+b] = t.inputs[copies[field][0]+b];
      mask[copies[field][1]+b] = 1;
    end
    predicted = result_item::type_id::create("expected");
    received = result_item::type_id::create("actual");
    predicted.value = want & mask;
    received.value = t.outputs & mask;
    expected.write(predicted);
    actual.write(received);
    begin
      if (`AI(t, ROCC_CMD_VALID) && `AO(t, ROCC_CMD_READY)) begin
        pending = 1;
        behavior.sample(0, 0, 0);
      end
      if (`AI(t, MOVE_COMPLETION_VALID) && `AO(t, MOVE_COMPLETION_READY)) pending = 0;
      if (`AI(t, ROCC_CMD_VALID) && !`AO(t, ROCC_CMD_READY)) behavior.sample(0, 1, 0);

    end
  endfunction
  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    if (pending || tags.size() != 0) `uvm_fatal("PENDING", "Admission ownership did not drain")
  endfunction
endclass
