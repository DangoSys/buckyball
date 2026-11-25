class ref_model extends reference_model #(request_item, response_item);
  `uvm_component_utils(ref_model)
  chandle model;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (`CACHE_ADDR_BITS > 64 || `CACHE_META_BITS > 32)
      `uvm_fatal("MODEL_CONFIG",
                 "DPI model supports addresses up to 64 bits and metadata up to 32 bits")
    model = cache_ref_create(`CACHE_SETS, `CACHE_WAYS, `CACHE_LINE_BYTES);
  endfunction
  function void write(request_item t);
    response_item expected = response_item::type_id::create("expected");
    byte unsigned hit, available, entry_valid;
    int unsigned way, metadata;
    longint unsigned addr;
    bit [`CACHE_LINE_BITS-1:0] data;
    cache_ref_access(model, t.op, t.addr, t.way, t.data, t.mask, t.metadata, t.eligible, hit,
                     available, way, entry_valid, addr, data, metadata);
    expected.id = t.id;
    expected.hit = hit;
    expected.available = available;
    expected.entry_valid = entry_valid;
    expected.way = way;
    expected.addr = addr;
    expected.data = data;
    expected.metadata = metadata;
    expected_ap.write(expected);
  endfunction
  function void final_phase(uvm_phase phase);
    cache_ref_destroy(model);
    super.final_phase(phase);
  endfunction
endclass
