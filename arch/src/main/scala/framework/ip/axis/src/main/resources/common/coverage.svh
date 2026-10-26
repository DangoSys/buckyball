class coverage extends uvm_subscriber #(item);
  `uvm_component_utils(coverage)
  covergroup behavior with function sample (logic [`AXIS_DATA_BITS/8-1:0] keep, bit last);
    option.per_instance = 1;
    masks: coverpoint keep {
      bins none = {0};
      bins full = {(1 << (`AXIS_DATA_BITS / 8)) - 1};
      bins partial = {[1 : (1 << (`AXIS_DATA_BITS / 8)) - 2]};
    }
    packet_boundary: coverpoint last;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    behavior = new();
  endfunction
  function void write(item t);
    behavior.sample(t.keep, t.last);
  endfunction
endclass
