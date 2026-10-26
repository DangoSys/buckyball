class coverage extends uvm_subscriber #(request_item);
  `uvm_component_utils(coverage)

  covergroup accesses with function sample (request_item req);
    option.per_instance = 1;
    direction: coverpoint req.write;
    address: coverpoint req.addr {bins rows[] = {[0 : `BANK_ROWS - 1]};}
    write_mask: coverpoint req.mask iff (req.write) {bins masks[] = {[0 : 15]};}
    direction_address: cross direction, address;
  endgroup

  function new(string name, uvm_component parent);
    super.new(name, parent);
    accesses = new();
  endfunction

  function void write(request_item req);
    accesses.sample(req);
  endfunction
endclass
