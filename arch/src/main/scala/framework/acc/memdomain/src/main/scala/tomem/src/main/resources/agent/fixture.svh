class fixture extends uvm_object;
  `uvm_object_utils(fixture)
  int groups[`TM_BANKS];
  bit read_error, write_error, bank_error, image_fault;
  int admission_error, completion_error;
  longint unsigned physical_offset = 'h100000;
  function new(string name = "fixture");
    super.new(name);
    foreach (groups[b]) groups[b] = 2;
  endfunction
  function byte unsigned initial_byte(longint unsigned address);
    return 8'(address ^ (address >> 8) ^ (address >> 16));
  endfunction
  function bit [127:0] initial_bank(int bank_id, int group_index, int row);
    bit [127:0] value;
    for (int b = 0; b < 16; b++) value[b*8+:8] = 8'(bank_id * 37 + group_index * 19 + row * 7 + b);
    return value;
  endfunction
endclass
