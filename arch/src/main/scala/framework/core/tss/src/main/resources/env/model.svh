class model extends uvm_subscriber #(cycle_item);
  `uvm_component_utils(model)
  uvm_analysis_port #(result_item) results;
  byte unsigned memory[`TSS_BYTES];
  bit initialized[`TSS_BYTES];
  bit pending[`TSS_PORTS], draining[`TSS_PORTS];
  int overtakes[`TSS_PORTS];
  const int offsets[`TSS_PORTS][13] = `TSS_PORT_OFFSETS;
  longint unsigned accepted, completed, cancelled;
  covergroup behavior with function sample (
      int port,
      bit writing,
      int size,
      int masking,
      bit error,
      bit cancelling,
      bit in_use,
      bit holding,
      bit contending
  );
    option.per_instance = 1;
    ports: coverpoint port {bins all[] = {[0 : `TSS_PORTS - 1]};}
    direction: coverpoint writing;
    sizes: coverpoint size {
      bins all[] = {[0 : $clog2(`TSS_MASK_BITS)]};
      ignore_bins invalid = {[$clog2(`TSS_MASK_BITS) + 1 : (1 << `TSS_SIZE_BITS) - 1]};
    }
    masks: coverpoint masking {bins all[] = {[0 : 2]};}
    errors: coverpoint error;
    cancels: coverpoint cancelling;
    usage: coverpoint in_use;
    stalls: coverpoint holding;
    contention: coverpoint contending;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    results  = new("results", this);
    behavior = new();
  endfunction
  function void event_result(int port, int kind, logic [`TSS_DATA_BITS-1:0] value = '0,
                             bit error = 0);
    result_item t = result_item::type_id::create("result");
    t.port = port;
    t.message_kind = kind;
    t.value = value;
    t.error = error;
    results.write(t);
  endfunction
  function void write(cycle_item t);
    int fires, offers;
    if (t.reset) begin
      foreach (pending[p]) begin
        if (pending[p]) cancelled++;
        event_result(p, 2);
        pending[p]   = 0;
        draining[p]  = 0;
        overtakes[p] = 0;
      end
      return;
    end
    offers = 0;
    fires  = 0;
    for (int p = 0; p < `TSS_PORTS; p++) begin
      if (t.inputs[offsets[p][0]]) offers++;
      if (t.inputs[offsets[p][0]] && t.outputs[offsets[p][6]]) fires++;
    end
    if (fires > 1) `uvm_fatal("ARBITRATION", "Multiple memory requests accepted in one cycle")
    for (int p = 0; p < `TSS_PORTS; p++) begin
      bit cancel_port, valid, ready, rvalid, rready, busy, in_use, eligible;
      cancel_port = p < `TSS_CLIENTS && t.inputs[offsets[p][12]];
      in_use = t.inputs[`TSS_IN_INUSE_OFFSET];
      valid = t.inputs[offsets[p][0]];
      ready = t.outputs[offsets[p][6]];
      rvalid = t.outputs[offsets[p][7]];
      rready = t.inputs[offsets[p][10]];
      busy = t.outputs[offsets[p][11]];
      verification_pkg::evidence::checks += 6;
      if (pending[p] && !busy) `uvm_fatal("BUSY", "An outstanding request lost its busy indication")
      if (!pending[p] && !draining[p] && busy) `uvm_fatal("BUSY", "An idle port is marked busy")
      if (draining[p] && !busy) draining[p] = 0;
      if ((cancel_port || (p < `TSS_CLIENTS && !in_use)) && ready)
        `uvm_fatal("GATING", "Disabled client accepted a request")
      if (draining[p] && (ready || rvalid))
        `uvm_fatal("CANCEL", "A cancelled request became visible")
      if (cancel_port && pending[p]) begin
        event_result(p, 2);
        pending[p]  = 0;
        draining[p] = 1;
        cancelled++;
        behavior.sample(p, 0, 0, 0, 0, 1, in_use, 0, offers > 1);
      end
      if (rvalid && !pending[p]) `uvm_fatal("RESPONSE", "Response without an outstanding request")
      if (rvalid && rready) begin
        event_result(p, 1, t.outputs[offsets[p][8]+:`TSS_DATA_BITS], t.outputs[offsets[p][9]]);
        pending[p] = 0;
        completed++;
      end
      eligible = valid && !busy && !cancel_port && (p == `TSS_CLIENTS || in_use);
      if (!eligible || ready) overtakes[p] = 0;
      else if (fires != 0) begin
        overtakes[p]++;
        if (overtakes[p] >= `TSS_PORTS)
          `uvm_fatal("FAIRNESS",
                     "Continuously eligible client was overtaken by a full arbitration round")
      end
      if (valid && ready) begin
        logic [63:0] address;
        logic [64:0] finish_address, limit_address;
        logic [`TSS_DATA_BITS-1:0] value, data;
        logic [`TSS_MASK_BITS-1:0] mask, allowed;
        int size, count, start;
        bit writing, legal;
        if (pending[p] || draining[p])
          `uvm_fatal("OUTSTANDING", "Port accepted a second outstanding request")
        address = t.inputs[offsets[p][1]+:64];
        size = t.inputs[offsets[p][2]+:`TSS_SIZE_BITS];
        count = 1 << size;
        writing = t.inputs[offsets[p][3]];
        data = t.inputs[offsets[p][4]+:`TSS_DATA_BITS];
        mask = t.inputs[offsets[p][5]+:`TSS_MASK_BITS];
        allowed = `TSS_MASK_BITS'((64'd1 << count) - 1);
        finish_address = {1'b0, address} + count;
        limit_address = {1'b0, `TSS_BASE} + `TSS_BYTES;
        legal=count<=`TSS_MASK_BITS && (address&(count-1))==0 && address>=`TSS_BASE && finish_address<=limit_address && !(writing&&((mask&~allowed)!=0));
        value = '0;
        if (legal) begin
          start = int'(address - `TSS_BASE);
          for (int b = 0; b < count; b++) begin
            if (writing) begin
              if (mask[b]) begin
                memory[start+b] = data[b*8+:8];
                initialized[start+b] = 1;
              end
            end else begin
              if (!initialized[start+b]) `uvm_fatal("REFERENCE", "Read of an uninitialized byte")
              value[b*8+:8] = memory[start+b];
            end
          end
        end
        event_result(p, 0, value, !legal);
        pending[p] = 1;
        accepted++;
        behavior.sample(p, writing, size, mask == 0 ? 0 : (mask == allowed ? 1 : 2), !legal, 0,
                        in_use, rvalid && !rready, offers > 1);
      end else if (rvalid && !rready) behavior.sample(p, 0, 0, 0, 0, 0, in_use, 1, offers > 1);
    end
  endfunction
  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    if (accepted != completed + cancelled)
      `uvm_fatal("CONSERVATION", "Accepted requests did not retire or cancel")
    foreach (pending[p])
    if (pending[p] || draining[p]) `uvm_fatal("PENDING", "Port has not drained")
  endfunction
endclass
