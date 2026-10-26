class coverage extends uvm_subscriber #(request_item);
  `uvm_component_utils(coverage)
  covergroup behavior with function sample (request_item t);
    option.per_instance = 1;
    operation: coverpoint t.write;
    size: coverpoint t.size {
      bins legal[] = {[0 : $clog2(`SPM_BITS / 8)]};
      bins oversized = {[$clog2(`SPM_BITS / 8) + 1 : $]};
    }
    lower_bound: coverpoint (t.address < `SPM_BASE) {bins seen = {1};}
    upper_bound: coverpoint (t.address >= `SPM_BASE + `SPM_BYTES) {bins seen = {1};}
    unaligned: coverpoint ((t.address & ((64'd1 << t.size) - 1)) != 0);
    read_only: coverpoint t.read_only;
    protected_write: coverpoint (t.read_only && t.write) {bins seen = {1};}
    empty_mask: coverpoint (t.mask == 0) iff (t.write);
    full_mask: coverpoint (&t.mask) iff (t.write);
    backpressure: coverpoint t.stall {bins stalls[] = {0, 3};}
    cancellation: coverpoint t.cancel;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    behavior = new();
  endfunction
  function void write(request_item t);
    behavior.sample(t);
  endfunction
endclass
