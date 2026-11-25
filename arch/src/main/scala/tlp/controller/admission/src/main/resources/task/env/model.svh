class model extends uvm_subscriber #(cycle_item);
  `uvm_component_utils(model)
  uvm_analysis_port #(result_item) expected, actual;
  const int copies[`AD_PAYLOAD_FIELDS][4] = `AD_PAYLOAD_MAP;
  bit pending, moving, has_result, work_done, response_done, move_error, remote_owner;
  int saved_tag, destination, tags[$];
  covergroup behavior with function sample (int route, bit stalled, bit cancelled);
    option.per_instance = 1;
    routes: coverpoint route {bins all[] = {[0 : 1]};}
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
      bit remote_command;
      logic [31:0] instruction;
      instruction = `AI(t, COMMAND_BITS_INSTRUCTION_RAW_INST);
      remote_command = instruction[31:25] >= 13;
      want[`AD_OUT_COMMAND_READY_OFFSET+:`AD_OUT_COMMAND_READY_WIDTH]=!pending&&(remote_command?
      `AI(t, REMOTE_REQUEST_READY)
      :
      `AI(t, LOCAL_REQUEST_READY)
      );
      mask[`AD_OUT_COMMAND_READY_OFFSET+:`AD_OUT_COMMAND_READY_WIDTH] = '1;

      want[`AD_OUT_LOCAL_REQUEST_VALID_OFFSET+:`AD_OUT_LOCAL_REQUEST_VALID_WIDTH] =
      `AI(t, COMMAND_VALID)
      && !pending && !remote_command;
      mask[`AD_OUT_LOCAL_REQUEST_VALID_OFFSET+:`AD_OUT_LOCAL_REQUEST_VALID_WIDTH] = '1;

      want[`AD_OUT_REMOTE_REQUEST_VALID_OFFSET+:`AD_OUT_REMOTE_REQUEST_VALID_WIDTH] =
      `AI(t, COMMAND_VALID)
      && !pending && remote_command;
      mask[`AD_OUT_REMOTE_REQUEST_VALID_OFFSET+:`AD_OUT_REMOTE_REQUEST_VALID_WIDTH] = '1;

      want[`AD_OUT_RESPONSE_VALID_OFFSET+:`AD_OUT_RESPONSE_VALID_WIDTH] = pending && (remote_owner ?
      `AI(t, REMOTE_REPLY_VALID)
      :
      `AI(t, LOCAL_REPLY_VALID)
      );
      mask[`AD_OUT_RESPONSE_VALID_OFFSET+:`AD_OUT_RESPONSE_VALID_WIDTH] = '1;

      if (want[`AD_OUT_RESPONSE_VALID_OFFSET]) begin
        want[`AD_OUT_RESPONSE_BITS_RD_OFFSET+:`AD_OUT_RESPONSE_BITS_RD_WIDTH] = destination;
        mask[`AD_OUT_RESPONSE_BITS_RD_OFFSET+:`AD_OUT_RESPONSE_BITS_RD_WIDTH] = '1;
        want[`AD_OUT_RESPONSE_BITS_DATA_OFFSET+:`AD_OUT_RESPONSE_BITS_DATA_WIDTH] = remote_owner ?
        `AI(t, REMOTE_REPLY_BITS)
        :
        `AI(t, LOCAL_REPLY_BITS);
        mask[`AD_OUT_RESPONSE_BITS_DATA_OFFSET+:`AD_OUT_RESPONSE_BITS_DATA_WIDTH] = '1;
      end
      want[`AD_OUT_LOCAL_REPLY_READY_OFFSET+:`AD_OUT_LOCAL_REPLY_READY_WIDTH]=pending&&!remote_owner&&`AI(
          t, RESPONSE_READY);
      mask[`AD_OUT_LOCAL_REPLY_READY_OFFSET+:`AD_OUT_LOCAL_REPLY_READY_WIDTH] = '1;

      want[`AD_OUT_REMOTE_REPLY_READY_OFFSET+:`AD_OUT_REMOTE_REPLY_READY_WIDTH]=pending&&remote_owner&&`AI(
          t, RESPONSE_READY);
      mask[`AD_OUT_REMOTE_REPLY_READY_OFFSET+:`AD_OUT_REMOTE_REPLY_READY_WIDTH] = '1;

      want[`AD_OUT_BUSY_OFFSET+:`AD_OUT_BUSY_WIDTH] = pending ||
      `AI(t, LOCAL_BUSY)
      ||
      `AI(t, REMOTE_BUSY);
      mask[`AD_OUT_BUSY_OFFSET+:`AD_OUT_BUSY_WIDTH] = '1;


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
      logic [31:0] instruction;
      bit remote_command;
      instruction = `AI(t, COMMAND_BITS_INSTRUCTION_RAW_INST);
      remote_command = instruction[31:25] >= 13;
      if (`AI(t, COMMAND_VALID) && `AO(t, COMMAND_READY)) begin
        pending = 1;
        remote_owner = remote_command;
        destination = instruction[11:7];
        behavior.sample(remote_command, 0, 0);
      end
      if (`AO(t, RESPONSE_VALID) && `AI(t, RESPONSE_READY)) pending = 0;
      if (
          `AI(t, COMMAND_VALID)
          && !
          `AO(t, COMMAND_READY)
          ||
          `AO(t, RESPONSE_VALID)
          && !
          `AI(t, RESPONSE_READY)
          )
        behavior.sample(remote_owner, 1, 0);

    end
  endfunction
  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    if (pending || tags.size() != 0) `uvm_fatal("PENDING", "Admission ownership did not drain")
  endfunction
endclass
