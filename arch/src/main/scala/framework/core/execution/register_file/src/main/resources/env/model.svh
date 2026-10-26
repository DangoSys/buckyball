class model extends reference_model #(request_item, response_item);
  `uvm_component_utils(model)
  logic [63:0] registers[32];
  bit initialized[32];
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void write(request_item req);
    response_item rsp = response_item::type_id::create("expected");
    if (req.write && req.address != 0) begin
      registers[req.address]   = req.data;
      initialized[req.address] = 1;
    end
    if ((req.read0 != 0 && !initialized[req.read0]) || (req.read1 != 0 && !initialized[req.read1]))
      `uvm_fatal("UNINITIALIZED", "Reading a register before its first write")
    rsp.data0 = req.read0 == 0 ? 0 : registers[req.read0];
    rsp.data1 = req.read1 == 0 ? 0 : registers[req.read1];
    expected_ap.write(rsp);
  endfunction
endclass
