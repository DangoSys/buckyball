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
      bit management, available;
      logic [31:0] instruction;
      instruction = `AI(t, EXECUTION_COMMAND_BITS_INSTRUCTION_RAW_INST);
      management = instruction[6:0] == 'h2b;
      available = !pending && !
      `AI(t, REMOTE_BUSY)
      && tags.size() < `AD_ENTRIES && (!management || (!
      `AI(t, ACCELERATOR_BUSY)
      && tags.size() == 0));
      want[`AD_OUT_EXECUTION_COMMAND_READY_OFFSET+:`AD_OUT_EXECUTION_COMMAND_READY_WIDTH]=available&&(management?
      `AI(t, REMOTE_REQUEST_READY)
      :
      `AI(t, ACCELERATOR_CMD_READY)
      );
      mask[`AD_OUT_EXECUTION_COMMAND_READY_OFFSET+:`AD_OUT_EXECUTION_COMMAND_READY_WIDTH] = '1;

      want[`AD_OUT_ACCELERATOR_CMD_VALID_OFFSET+:`AD_OUT_ACCELERATOR_CMD_VALID_WIDTH] =
      `AI(t, EXECUTION_COMMAND_VALID)
      && available && !management;
      mask[`AD_OUT_ACCELERATOR_CMD_VALID_OFFSET+:`AD_OUT_ACCELERATOR_CMD_VALID_WIDTH] = '1;

      want[`AD_OUT_REMOTE_REQUEST_VALID_OFFSET+:`AD_OUT_REMOTE_REQUEST_VALID_WIDTH] =
      `AI(t, EXECUTION_COMMAND_VALID)
      && available && management;
      mask[`AD_OUT_REMOTE_REQUEST_VALID_OFFSET+:`AD_OUT_REMOTE_REQUEST_VALID_WIDTH] = '1;

      want[`AD_OUT_ACCELERATOR_RESP_READY_OFFSET+:`AD_OUT_ACCELERATOR_RESP_READY_WIDTH] = 0;
      mask[`AD_OUT_ACCELERATOR_RESP_READY_OFFSET+:`AD_OUT_ACCELERATOR_RESP_READY_WIDTH] = '1;

      want[`AD_OUT_EXECUTION_RESPONSE_VALID_OFFSET+:`AD_OUT_EXECUTION_RESPONSE_VALID_WIDTH]=pending&&!work_done&&has_result&&`AI(
          t, REMOTE_REPLY_VALID);
      mask[`AD_OUT_EXECUTION_RESPONSE_VALID_OFFSET+:`AD_OUT_EXECUTION_RESPONSE_VALID_WIDTH] = '1;

      if (want[`AD_OUT_EXECUTION_RESPONSE_VALID_OFFSET]) begin
        want[`AD_OUT_EXECUTION_RESPONSE_BITS_RD_OFFSET+:`AD_OUT_EXECUTION_RESPONSE_BITS_RD_WIDTH]=destination;
        mask[`AD_OUT_EXECUTION_RESPONSE_BITS_RD_OFFSET+:`AD_OUT_EXECUTION_RESPONSE_BITS_RD_WIDTH]='1;
        want[`AD_OUT_EXECUTION_RESPONSE_BITS_DATA_OFFSET+:`AD_OUT_EXECUTION_RESPONSE_BITS_DATA_WIDTH]=`AI(
            t, REMOTE_REPLY_BITS);
        mask[`AD_OUT_EXECUTION_RESPONSE_BITS_DATA_OFFSET+:`AD_OUT_EXECUTION_RESPONSE_BITS_DATA_WIDTH]='1;
      end
      want[`AD_OUT_REMOTE_REPLY_READY_OFFSET+:`AD_OUT_REMOTE_REPLY_READY_WIDTH]=pending&&!work_done&&(!has_result||
      `AI(t, EXECUTION_RESPONSE_READY)
      );
      mask[`AD_OUT_REMOTE_REPLY_READY_OFFSET+:`AD_OUT_REMOTE_REPLY_READY_WIDTH] = '1;

      want[`AD_OUT_EXECUTION_COMPLETE_VALID_OFFSET+:`AD_OUT_EXECUTION_COMPLETE_VALID_WIDTH]=tags.size()!=0&&(pending?work_done:!
      `AI(t, ACCELERATOR_BUSY)
      );
      mask[`AD_OUT_EXECUTION_COMPLETE_VALID_OFFSET+:`AD_OUT_EXECUTION_COMPLETE_VALID_WIDTH] = '1;

      if (want[`AD_OUT_EXECUTION_COMPLETE_VALID_OFFSET]) begin
        want[`AD_OUT_EXECUTION_COMPLETE_BITS_TAG_OFFSET+:`AD_OUT_EXECUTION_COMPLETE_BITS_TAG_WIDTH]=tags[0];
        mask[`AD_OUT_EXECUTION_COMPLETE_BITS_TAG_OFFSET+:`AD_OUT_EXECUTION_COMPLETE_BITS_TAG_WIDTH]='1;
      end
      want[`AD_OUT_DRAINED_OFFSET+:`AD_OUT_DRAINED_WIDTH] = !pending && tags.size() == 0 && !
      `AI(t, ACCELERATOR_BUSY)
      && !
      `AI(t, REMOTE_BUSY);
      mask[`AD_OUT_DRAINED_OFFSET+:`AD_OUT_DRAINED_WIDTH] = '1;


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
      bit management;
      instruction = `AI(t, EXECUTION_COMMAND_BITS_INSTRUCTION_RAW_INST);
      management  = instruction[6:0] == 'h2b;
      if (`AO(t, EXECUTION_COMPLETE_VALID) && `AI(t, EXECUTION_COMPLETE_READY)) begin
        int tag = tags.pop_front();
        if (pending) pending = 0;
      end
      if (`AI(t, EXECUTION_COMMAND_VALID) && `AO(t, EXECUTION_COMMAND_READY)) begin
        tags.push_back(`AI(t, EXECUTION_COMMAND_BITS_TAG));
        if (management) begin
          pending = 1;
          work_done = 0;
          has_result = instruction[14];
          destination = instruction[11:7];
        end
        behavior.sample(management, 0, 0);
      end
      if (`AI(t, REMOTE_REPLY_VALID) && `AO(t, REMOTE_REPLY_READY)) work_done = 1;
      if (
          `AI(t, EXECUTION_COMMAND_VALID)
          && !
          `AO(t, EXECUTION_COMMAND_READY)
          ||
          `AO(t, EXECUTION_COMPLETE_VALID)
          && !
          `AI(t, EXECUTION_COMPLETE_READY)
          )
        behavior.sample(management, 1, 0);

    end
  endfunction
  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    if (pending || tags.size() != 0) `uvm_fatal("PENDING", "Admission ownership did not drain")
  endfunction
endclass
