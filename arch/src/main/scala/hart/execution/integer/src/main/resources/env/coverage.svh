class coverage extends uvm_subscriber #(request_item);
  `uvm_component_utils(coverage)
  covergroup behavior with function sample (request_item t);
    option.per_instance = 1;
    operation: coverpoint t.fn {
      bins arithmetic[] = {0, 10};
      bins shift[] = {1, 5, 11, 17, 18, 19};
      bins logic_ops[] = {4, 6, 7, 8, 9, 24, 25, 26};
      bins compare_ops[] = {12, 13, 14, 15, 28, 29, 30, 31};
      bins unary = {16};
    }
    width: coverpoint t.dw;
    unary: coverpoint t.rhs[11:0] iff (t.fn == 16) {
      bins selectors[] = {0, 1, 2, 'h287, 'h6b8, 'h080, 'h604, 'h605};
    }
    shift_amount: coverpoint (t.rhs & (t.dw ? 63 : 31)) iff (t.fn inside {1, 5, 11, 17, 18, 19}) {
      bins boundaries[] = {0, 1, 31, 32, 63};
    }
    sign_a: coverpoint t.a[63];
    sign_rhs: coverpoint t.rhs[63];
    operand_signs: cross sign_a, sign_rhs;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    behavior = new();
  endfunction
  function void write(request_item t);
    behavior.sample(t);
  endfunction
endclass
