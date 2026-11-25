class coverage extends uvm_subscriber #(request_item);
  `uvm_component_utils(coverage)
  covergroup operations with function sample (request_item req);
    option.per_instance = 1;
    operation: coverpoint req.op {bins supported[] = {[0 : 4]};}
    set_index: coverpoint ((req.addr / `CACHE_LINE_BYTES) % `CACHE_SETS) {
      bins sets[] = {[0 : `CACHE_SETS - 1]};
    }
    way: coverpoint req.way {bins ways[] = {[0 : `CACHE_WAYS - 1]};}
    mask: coverpoint req.mask iff (req.op == WRITE) {
      bins none = {'0}; bins all = {'1}; bins partial = default;
    }
    eligibility: coverpoint (req.eligible != 0);
    operation_set: cross operation, set_index;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    operations = new();
  endfunction
  function void write(request_item t);
    operations.sample(t);
  endfunction
endclass
