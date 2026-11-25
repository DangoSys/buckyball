class model extends reference_model #(request_item, response_item);
  `uvm_component_utils(model)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void write(request_item req);
    response_item rsp = response_item::type_id::create("expected");
    bit signed_a = req.operation < 4 ? req.operation inside {1, 2} : !req.operation[0];
    bit signed_b = req.operation < 4 ? req.operation == 1 : !req.operation[0];
    logic [63:0] a = req.word ? (signed_a ? {{32{req.a[31]}},req.a[31:0]} : {32'b0,req.a[31:0]}) : req.a;
    logic [63:0] b = req.word ? (signed_b ? {{32{req.rhs[31]}},req.rhs[31:0]} : {32'b0,req.rhs[31:0]}) : req.rhs;
    logic signed [128:0] wide_a = signed_a ? $signed({{65{a[63]}}, a}) : $signed({65'b0, a});
    logic signed [128:0] wide_b = signed_b ? $signed({{65{b[63]}}, b}) : $signed({65'b0, b});
    logic signed [257:0] product = wide_a * wide_b;
    logic [63:0] answer;
    if (req.operation < 4) begin
      answer = req.operation == 0 ? product[63:0] : product[127:64];
    end else if (b == 0) begin
      answer = req.operation[1] ? a : '1;
    end else if (!req.operation[0] && a == 64'h8000000000000000 && b == '1) begin
      answer = req.operation[1] ? 0 : a;
    end else begin
      case (req.operation)
        4: answer = $signed(a) / $signed(b);
        5: answer = a / b;
        6: answer = $signed(a) % $signed(b);
        7: answer = a % b;
        default: `uvm_fatal("OPERATION", "Invalid RV64M operation")
      endcase
    end
    rsp.value = req.word ? {{32{answer[31]}}, answer[31:0]} : answer;
    expected_ap.write(rsp);
  endfunction
endclass
