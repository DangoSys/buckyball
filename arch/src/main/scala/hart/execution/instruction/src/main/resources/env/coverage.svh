class coverage extends uvm_subscriber #(request_item);
  `uvm_component_utils(coverage)
  covergroup behavior with function sample (request_item t);
    option.per_instance = 1;
    quadrant: coverpoint t.bits[1:0] {bins compressed[] = {[0 : 2]};}
    compressed_opcode: coverpoint t.bits[15:13] iff (t.bits[1:0] != 3) {
      bins functions[] = {[0 : 7]};
    }
    compressed_encodings: cross quadrant, compressed_opcode;
    opcode: coverpoint t.bits[6:0] iff (t.bits[1:0] == 3) {
      bins opcodes[]={7'h03,7'h07,7'h0f,7'h13,7'h17,7'h1b,7'h23,7'h27,7'h2b,7'h2f,7'h33,7'h37,7'h3b,7'h43,7'h47,7'h4b,7'h4f,7'h53,7'h63,7'h67,7'h6f,7'h73,7'h7b};
    }
    rd: coverpoint t.bits[11:7] {bins registers[] = {[0 : 31]};}
    rs1: coverpoint t.bits[19:15] iff (t.bits[1:0] == 3) {bins registers[] = {[0 : 31]};}
    rs2: coverpoint t.bits[24:20] iff (t.bits[1:0] == 3) {bins registers[] = {[0 : 31]};}
    immediate_sign: coverpoint t.bits[31] iff (t.bits[1:0] == 3);
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    behavior = new();
  endfunction
  function void write(request_item t);
    behavior.sample(t);
  endfunction
endclass
