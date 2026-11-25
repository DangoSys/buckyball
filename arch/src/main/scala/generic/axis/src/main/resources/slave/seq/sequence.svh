class protocol_sequence extends uvm_sequence #(item);
  `uvm_object_utils(protocol_sequence)

  function new(string name = "protocol_sequence");
    super.new(name);
  endfunction

  task body();
    item req;
    for (int i = 0; i < 32; i++) begin
      req = item::type_id::create($sformatf("item_%0d", i));
      start_item(req);
      req.data = $urandom();
      req.keep = i % (1 << (`AXIS_DATA_BITS / 8));
      req.last = (i % 4) == 3;
      finish_item(req);
    end
  endtask
endclass
