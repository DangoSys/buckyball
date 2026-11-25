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
      bit offered, task_command, move_command, write_result;
      logic [31:0] instruction;
      instruction = `AI(t, EXECUTION_COMMAND_BITS_INSTRUCTION_RAW_INST);
      offered = `AI(t, EXECUTION_COMMAND_VALID);
      task_command = instruction[6:0] == 'h2b;
      move_command = instruction[6:0] == 'h7b && instruction[31:25] == 'h0d;
      want[`AD_OUT_EXECUTION_COMMAND_READY_OFFSET+:`AD_OUT_EXECUTION_COMMAND_READY_WIDTH]=!pending&&(task_command?
      `AI(t, TASKREQUEST_READY)
      : (move_command &&
      `AI(t, MOVE_COMMAND_READY)
      ));
      mask[`AD_OUT_EXECUTION_COMMAND_READY_OFFSET+:`AD_OUT_EXECUTION_COMMAND_READY_WIDTH] = '1;

      want[`AD_OUT_TASKREQUEST_VALID_OFFSET+:`AD_OUT_TASKREQUEST_VALID_WIDTH]=!pending&&offered&&task_command;
      mask[`AD_OUT_TASKREQUEST_VALID_OFFSET+:`AD_OUT_TASKREQUEST_VALID_WIDTH] = '1;

      want[`AD_OUT_MOVE_COMMAND_VALID_OFFSET+:`AD_OUT_MOVE_COMMAND_VALID_WIDTH]=!pending&&offered&&move_command;
      mask[`AD_OUT_MOVE_COMMAND_VALID_OFFSET+:`AD_OUT_MOVE_COMMAND_VALID_WIDTH] = '1;

      want[`AD_OUT_MOVE_COMMAND_BITS_ROWS_OFFSET+:`AD_OUT_MOVE_COMMAND_BITS_ROWS_WIDTH]={1'b0,t.inputs[`AD_IN_EXECUTION_COMMAND_BITS_INSTRUCTION_RS2DATA_OFFSET+32+:16]}+17'd1;
      mask[`AD_OUT_MOVE_COMMAND_BITS_ROWS_OFFSET+:`AD_OUT_MOVE_COMMAND_BITS_ROWS_WIDTH] = '1;

      want[`AD_OUT_MOVE_COMPLETION_READY_OFFSET+:`AD_OUT_MOVE_COMPLETION_READY_WIDTH]=pending&&moving&&!work_done;
      mask[`AD_OUT_MOVE_COMPLETION_READY_OFFSET+:`AD_OUT_MOVE_COMPLETION_READY_WIDTH] = '1;

      want[`AD_OUT_TASKRESPONSE_READY_OFFSET+:`AD_OUT_TASKRESPONSE_READY_WIDTH]=pending&&!moving&&!work_done&&(!has_result||
      `AI(t, EXECUTION_RESPONSE_READY)
      );
      mask[`AD_OUT_TASKRESPONSE_READY_OFFSET+:`AD_OUT_TASKRESPONSE_READY_WIDTH] = '1;

      want[`AD_OUT_EXECUTION_RESPONSE_VALID_OFFSET+:`AD_OUT_EXECUTION_RESPONSE_VALID_WIDTH]=pending&&has_result&&!response_done&&(moving?work_done:
      `AI(t, TASKRESPONSE_VALID)
      );
      mask[`AD_OUT_EXECUTION_RESPONSE_VALID_OFFSET+:`AD_OUT_EXECUTION_RESPONSE_VALID_WIDTH] = '1;

      if (want[`AD_OUT_EXECUTION_RESPONSE_VALID_OFFSET]) begin
        want[`AD_OUT_EXECUTION_RESPONSE_BITS_RD_OFFSET+:`AD_OUT_EXECUTION_RESPONSE_BITS_RD_WIDTH]=moving?destination:`AI(
            t, TASKRESPONSE_BITS_RD);
        mask[`AD_OUT_EXECUTION_RESPONSE_BITS_RD_OFFSET+:`AD_OUT_EXECUTION_RESPONSE_BITS_RD_WIDTH]='1;

        want[`AD_OUT_EXECUTION_RESPONSE_BITS_DATA_OFFSET+:`AD_OUT_EXECUTION_RESPONSE_BITS_DATA_WIDTH]=moving?64'(move_error):`AI(
            t, TASKRESPONSE_BITS_DATA);
        mask[`AD_OUT_EXECUTION_RESPONSE_BITS_DATA_OFFSET+:`AD_OUT_EXECUTION_RESPONSE_BITS_DATA_WIDTH]='1;

      end
      want[`AD_OUT_EXECUTION_COMPLETE_VALID_OFFSET+:`AD_OUT_EXECUTION_COMPLETE_VALID_WIDTH]=pending&&work_done&&(!has_result||response_done);
      mask[`AD_OUT_EXECUTION_COMPLETE_VALID_OFFSET+:`AD_OUT_EXECUTION_COMPLETE_VALID_WIDTH] = '1;

      if (want[`AD_OUT_EXECUTION_COMPLETE_VALID_OFFSET]) begin
        want[`AD_OUT_EXECUTION_COMPLETE_BITS_TAG_OFFSET+:`AD_OUT_EXECUTION_COMPLETE_BITS_TAG_WIDTH]=saved_tag;
        mask[`AD_OUT_EXECUTION_COMPLETE_BITS_TAG_OFFSET+:`AD_OUT_EXECUTION_COMPLETE_BITS_TAG_WIDTH]='1;
      end
      want[`AD_OUT_MEMORYALLOWED_OFFSET+:`AD_OUT_MEMORYALLOWED_WIDTH]=!(offered&&move_command||pending&&(moving||work_done));
      mask[`AD_OUT_MEMORYALLOWED_OFFSET+:`AD_OUT_MEMORYALLOWED_WIDTH] = '1;

      want[`AD_OUT_IDLE_OFFSET+:`AD_OUT_IDLE_WIDTH] = !pending;
      mask[`AD_OUT_IDLE_OFFSET+:`AD_OUT_IDLE_WIDTH] = '1;


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
      bit task_command, move_command;
      instruction  = `AI(t, EXECUTION_COMMAND_BITS_INSTRUCTION_RAW_INST);
      task_command = instruction[6:0] == 'h2b;
      move_command = instruction[6:0] == 'h7b && instruction[31:25] == 'h0d;
      if (`AI(t, EXECUTION_COMMAND_VALID) && `AO(t, EXECUTION_COMMAND_READY)) begin
        pending = 1;
        moving = move_command;
        has_result = instruction[14];
        destination = instruction[11:7];
        saved_tag = `AI(t, EXECUTION_COMMAND_BITS_TAG);
        work_done = 0;
        response_done = 0;
        behavior.sample(move_command, 0, 0);
      end
      if (`AI(t, TASKRESPONSE_VALID) && `AO(t, TASKRESPONSE_READY)) work_done = 1;
      if (`AI(t, MOVE_COMPLETION_VALID) && `AO(t, MOVE_COMPLETION_READY)) begin
        work_done  = 1;
        move_error = `AI(t, MOVE_COMPLETION_BITS);
      end
      if (`AO(t, EXECUTION_RESPONSE_VALID) && `AI(t, EXECUTION_RESPONSE_READY)) response_done = 1;
      if (`AO(t, EXECUTION_COMPLETE_VALID) && `AI(t, EXECUTION_COMPLETE_READY)) pending = 0;
      if (
          `AI(t, EXECUTION_COMMAND_VALID)
          && !
          `AO(t, EXECUTION_COMMAND_READY)
          ||
          `AO(t, EXECUTION_COMPLETE_VALID)
          && !
          `AI(t, EXECUTION_COMPLETE_READY)
          ||
          `AO(t, EXECUTION_RESPONSE_VALID)
          && !
          `AI(t, EXECUTION_RESPONSE_READY)
          )
        behavior.sample(moving, 1, 0);

    end
  endfunction
  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    if (pending || tags.size() != 0) `uvm_fatal("PENDING", "Admission ownership did not drain")
  endfunction
endclass
