class coverage extends uvm_subscriber #(request_item);
  `uvm_component_utils(coverage)
  covergroup behavior with function sample (request_item t);
    option.per_instance = 1;
    read0: coverpoint t.read0 {bins addresses[] = {[0 : 31]};}
    read1: coverpoint t.read1 {bins addresses[] = {[0 : 31]};}
    read_pairs: cross read0, read1;
    write: coverpoint t.write;
    destination: coverpoint t.address iff (t.write) {bins addresses[] = {[0 : 31]};}
    bypass0: coverpoint (t.write && t.address != 0 && t.address == t.read0);
    bypass1: coverpoint (t.write && t.address != 0 && t.address == t.read1);
    write_zero: coverpoint (t.write && t.address == 0) {bins seen = {1};}
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    behavior = new();
  endfunction
  function void write(request_item t);
    behavior.sample(t);
  endfunction
endclass
