class fixture extends uvm_object;
  `uvm_object_utils(fixture)
  logic [63:0] ptes[longint unsigned];
  bit errors[longint unsigned];
  bit deny_pte, deny_data, backend_error;
  longint unsigned authorized_pte;
  function new(string name = "fixture");
    super.new(name);
  endfunction
  function logic [63:0] word(longint unsigned address);
    return 64'h80ff7f0181fe8000 ^ (address * 64'h0101010101010101);
  endfunction
  function logic [`VA_LINE_BITS-1:0] line(longint unsigned address);
    logic [`VA_LINE_BITS-1:0] value;
    for (int i = 0; i < `VA_LINE_BITS / 64; i++)
    value[i*64+:64] = word((address & ~64'd63) + i * 8);
    return value;
  endfunction
endclass
