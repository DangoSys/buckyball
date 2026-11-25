class coverage extends uvm_subscriber #(request_item);
  `uvm_component_utils(coverage)
  covergroup behavior with function sample (request_item t);
    option.per_instance = 1;
    operation: coverpoint t.operation {bins operations[] = {[0 : 7]};}
    word: coverpoint t.word;
    operation_width: cross operation, word{
      ignore_bins invalid_word = binsof (operation) intersect {[1 : 3]} && binsof (word) intersect {
        1
      };
    }
    zero_divisor: coverpoint (t.rhs == 0) iff (t.operation >= 4);
    signed_overflow: coverpoint (t.a == 64'h8000000000000000 && t.rhs == '1)
      iff (!t.word && t.operation inside {4,6}) {
      bins seen = {1};
    }
    word_overflow: coverpoint (t.a[31:0] == 32'h80000000 && t.rhs[31:0] == '1)
      iff (t.word && t.operation inside {4,6}) {
      bins seen = {1};
    }
    sign_a: coverpoint t.a[63];
    sign_rhs: coverpoint t.rhs[63];
    signs: cross sign_a, sign_rhs;
    backpressure: coverpoint t.stall {bins stalls[] = {0, 3};}
    cancelled: coverpoint t.cancel;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    behavior = new();
  endfunction
  function void write(request_item t);
    behavior.sample(t);
  endfunction
endclass
