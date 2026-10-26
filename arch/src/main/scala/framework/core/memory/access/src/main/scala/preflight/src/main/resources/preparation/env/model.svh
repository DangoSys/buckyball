class model extends uvm_subscriber #(cycle_item);
  `uvm_component_utils(model)
  fixture image;
  chandle reference;
  uvm_analysis_port #(result_item) expected, actual;
  uvm_analysis_port #(uvm_sequence_item) cancelled;
  int index[int], errors[int], counts[int];
  longint unsigned addresses[int][int], physical[int][int];
  int lengths[int][int];
  bit direction[int], notified[int], live[int];
  longint query_key;
  covergroup behavior with function sample (
      int error, bit writing, int mode, int policy, bit hit, bit released
  );
    option.per_instance = 1;
    outcomes: coverpoint error {
      bins success = {0};
      bins shape = {1};
      bins page_fault = {3};
      bins access_fault = {4};
      bins capacity = {5};
    }
    directions: coverpoint writing;
    modes: coverpoint mode {bins all[] = {0, 8};}
    pmp: coverpoint policy {bins all[] = {0, 1, 2, 3, 4, 6};}
    queries: coverpoint hit;
    release_events: coverpoint released;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    expected = new("expected", this);
    actual = new("actual", this);
    cancelled = new("cancelled", this);
    reference = preflight_ref_create(`PREP_ADDRESS_BITS, 16);
    behavior = new();
  endfunction
  function void verify(bit ok, string message);
    evidence::checks++;
    if (!ok) `uvm_fatal("PREPARATION", message)
  endfunction
  function void predict(cycle_item t);
    int id, policy, count;
    bit deny;
    int mode;
    longint unsigned base;
    id = `AI(t, COMMAND_BITS_ID);
    policy = image.policy[id];
    mode = `AI(t, COMMAND_BITS_MODE);
    base = `AI(t, COMMAND_BITS_BASEVA);
    deny = policy == 1 || policy == 4 || (policy == 2 && `AI(t, COMMAND_BITS_WRITE));
    preflight_ref_program(reference, 'h10000, 'hcf, image.pte_error);
    preflight_ref_policy(reference, deny && mode == 8, 'h10000, deny, 0,
                         (64'd1 << `PREP_ADDRESS_BITS) - 1);
    verify(preflight_ref_submit(
           reference,
           id,
           base,
           `AI(t, COMMAND_BITS_ROWS),
           `AI(t, COMMAND_BITS_COLUMNS),
           `AI(t, COMMAND_BITS_SPANBYTES),
           `AI(t, COMMAND_BITS_COLUMNSTRIDE),
           `AI(t, COMMAND_BITS_ROWSTRIDE),
           `AI(t, COMMAND_BITS_WRITE),
           mode,
           `AI(t, COMMAND_BITS_ROOTPPN),
           `AI(t, COMMAND_BITS_PRIVILEGE),
           `AI(t, COMMAND_BITS_SUM),
           `AI(t, COMMAND_BITS_MXR)
           ) == 0, "Missing preparation fixture");
    verify(!live.exists(id), "Duplicate live command ID");
    live[id] = 1;
    direction[id] = `AI(t, COMMAND_BITS_WRITE);
    index[id] = 0;
    count = preflight_ref_count(reference, id);
    counts[id] = count;
    for (int i = 0; i < count; i++) begin
      result_item result = result_item::type_id::create("range_expected");
      longint unsigned va, pa;
      int bytes, writing, last, error;
      preflight_ref_expected(reference, id, i, va, pa, bytes, writing, last, error);
      result.value = {32'(id), va, pa, 32'(bytes), 32'(writing), 32'(last), 32'(error)};
      result.set_transaction_id((id << 4) | i);
      expected.write(result);
      addresses[id][i] = va;
      physical[id][i]  = pa;
      lengths[id][i]   = bytes;
      if (last) begin
        result = result_item::type_id::create("ready_expected");
        result.value = {32'(id), 32'(error), error == 0 ? 64'd0 : va};
        result.set_transaction_id((1 << 20) | id);
        expected.write(result);
        errors[id] = error;
        behavior.sample(error, writing, mode, policy, 0, 0);
      end
    end
  endfunction
  function void query(cycle_item t, int port, int valid, int id, longint unsigned va, int bytes,
                      bit writing, bit hit, longint unsigned pa, int error);
    bit expected_hit, shape;
    longint unsigned expected_pa;
    int expected_error;
    logic [64:0] finish_address = {1'b0, va} + bytes;
    bit releasing = `AI(t, RELEASE_VALID) && `AO(t, RELEASE_READY) && `AI(t, RELEASE_BITS_ID) == id;
    shape = bytes > 0 && !finish_address[64] && int'(va & 4095) + bytes <= 4096;
    expected_hit = 0;
    expected_pa = 0;
    if (valid && shape && live.exists(
            id
        ) && notified.exists(
            id
        ) && errors[id] == 0 && !releasing && direction[id] == writing)
      for (int i = 0; i < counts[id]; i++)
      if (va >= addresses[id][i] && finish_address <= addresses[id][i] + lengths[id][i]) begin
        expected_hit = 1;
        expected_pa  = physical[id][i] + va - addresses[id][i];
      end
    expected_error = !valid || expected_hit ? 0 : (!shape ? 1 : 4);
    verify(hit == expected_hit && pa == expected_pa && error == expected_error, $sformatf(
           "Query port%0d id%0d VA%h expected hit%0d PA%h error%0d got hit%0d PA%h error%0d",
           port,
           id,
           va,
           expected_hit,
           expected_pa,
           expected_error,
           hit,
           pa,
           error
           ));
    if (valid) behavior.sample(expected_error, writing, 0, 0, hit, releasing);
  endfunction
  function void write(cycle_item t);
    if (t.reset) begin
      cancelled.write(null);
      preflight_ref_reset(reference);
      index.delete();
      errors.delete();
      counts.delete();
      addresses.delete();
      physical.delete();
      lengths.delete();
      direction.delete();
      notified.delete();
      live.delete();
      return;
    end
    query(t, 0, `AI(t, QUERIES_0_VALID), `AI(t, QUERIES_0_ID), `AI(t, QUERIES_0_VA),
          `AI(t, QUERIES_0_BYTES), `AI(t, QUERIES_0_WRITE), `AO(t, RESULTS_0_HIT),
          `AO(t, RESULTS_0_PA), `AO(t, RESULTS_0_ERROR));
    query(t, 1, `AI(t, QUERIES_1_VALID), `AI(t, QUERIES_1_ID), `AI(t, QUERIES_1_VA),
          `AI(t, QUERIES_1_BYTES), `AI(t, QUERIES_1_WRITE), `AO(t, RESULTS_1_HIT),
          `AO(t, RESULTS_1_PA), `AO(t, RESULTS_1_ERROR));
    if (`AI(t, COMMAND_VALID) && `AO(t, COMMAND_READY)) predict(t);
    if (`AO(t, PTEREQUEST_VALID) && `AI(t, PTEREQUEST_READY)) begin
      longint unsigned value;
      int error, id;
      id = `AO(t, CONTEXTTAG);
      verify(preflight_ref_pte(reference, id, `AO(t, PTEREQUEST_BITS_ADDR), value, error) == 1,
             "PTE request differs from preparation plan");
    end
    if (`AO(t, RANGES_VALID) && `AI(t, RANGES_READY)) begin
      int id = `AO(t, RANGES_BITS_ID);
      result_item result = result_item::type_id::create("range_actual");
      verify(live.exists(id), "Prepared range has no live command");
      result.value = {
        32'(id),
        64'(`AO(t, RANGES_BITS_VA)),
        64'(`AO(t, RANGES_BITS_PA)),
        32'(`AO(t, RANGES_BITS_BYTES)),
        32'(`AO(t, RANGES_BITS_WRITE)),
        32'(`AO(t, RANGES_BITS_LAST)),
        32'(`AO(t, RANGES_BITS_ERROR))
      };
      result.set_transaction_id((id << 4) | index[id]);
      actual.write(result);
      index[id]++;
    end
    if (`AO(t, READY_VALID) && `AI(t, READY_READY)) begin
      int id = `AO(t, READY_BITS_ID);
      result_item result = result_item::type_id::create("ready_actual");
      verify(live.exists(id) && index[id] == counts[id],
             "Notification before all ranges were retained");
      result.value = {32'(id), 32'(`AO(t, READY_BITS_ERROR)), 64'(`AO(t, READY_BITS_VA))};
      result.set_transaction_id((1 << 20) | id);
      actual.write(result);
      notified[id] = 1;
    end
    if (`AI(t, RELEASE_VALID) && `AO(t, RELEASE_READY)) begin
      int id = `AI(t, RELEASE_BITS_ID);
      verify(notified.exists(id), "Released a mapping before notification");
      preflight_ref_retire(reference, id);
      live.delete(id);
      notified.delete(id);
      behavior.sample(errors[id], direction[id], 0, 0, 0, 1);
    end
  endfunction
  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    verify(live.num() == 0, "Mappings did not release");
  endfunction
  function void final_phase(uvm_phase phase);
    preflight_ref_destroy(reference);
    super.final_phase(phase);
  endfunction
endclass
