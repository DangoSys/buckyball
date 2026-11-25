class model extends reference_model #(request_item, response_item);
  `uvm_component_utils(model)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void write(request_item req);
    response_item rsp = response_item::type_id::create("expected");
    logic [63:0] value;
    int width = req.dw ? 64 : 32;
    int shift = req.rhs & (width - 1);
    logic [63:0] operand = req.dw ? req.a : {32'b0, req.a[31:0]};
    int count;
    rsp.adder = req.fn[3] ? req.a - req.rhs : req.a + req.rhs;
    rsp.comparison = req.fn[0] ^ (!req.fn[3] ? req.a == req.rhs :
                                req.fn[1] ? req.a < req.rhs : $signed(req.a) < $signed(req.rhs));
    case (req.fn)
      0: value = req.a + req.rhs;
      10: value = req.a - req.rhs;
      1: value = operand << shift;
      5: value = operand >> shift;
      11: value = req.dw ? $signed(req.a) >>> shift : $signed(req.a[31:0]) >>> shift;
      4: value = req.a ^ req.rhs;
      6: value = req.a | req.rhs;
      7: value = req.a & req.rhs;
      8: value = req.rhs != 0 ? req.a : 0;
      9: value = req.rhs == 0 ? req.a : 0;
      12, 13: value = $signed(req.a) < $signed(req.rhs);
      14, 15: value = req.a < req.rhs;
      17: value = (operand << shift) | (operand >> (width - shift));
      18: value = (operand >> shift) | (operand << (width - shift));
      19: value = (operand >> shift) & 1;
      24: value = req.a & ~req.rhs;
      25: value = req.a | ~req.rhs;
      26: value = req.a ^ ~req.rhs;
      28: value = $signed(req.a) > $signed(req.rhs) ? req.a : req.rhs;
      29: value = $signed(req.a) < $signed(req.rhs) ? req.a : req.rhs;
      30: value = req.a > req.rhs ? req.a : req.rhs;
      31: value = req.a < req.rhs ? req.a : req.rhs;
      16: begin
        value = 0;
        case (req.rhs[11:0])
          12'h287: for (int i = 0; i < 8; i++) value[i*8+:8] = |req.a[i*8+:8] ? 8'hff : 0;
          12'h6b8: for (int i = 0; i < 8; i++) value[i*8+:8] = req.a[(7-i)*8+:8];
          12'h080: value = {48'b0, req.a[15:0]};
          12'h604: value = {{56{req.a[7]}}, req.a[7:0]};
          12'h605: value = {{48{req.a[15]}}, req.a[15:0]};
          0, 1, 2: begin
            count = 0;
            if (req.rhs == 2) begin
              for (int i = 0; i < width; i++) count += operand[i];
            end else begin
              while (count < width && !operand[req.rhs==0?width-1-count : count]) count++;
            end
            value = count;
          end
          default: `uvm_fatal("OPERATION", "Unknown unary instruction")
        endcase
      end
      default: `uvm_fatal("OPERATION", "Unknown integer operation")
    endcase
    rsp.value = req.dw ? value : {{32{value[31]}}, value[31:0]};
    expected_ap.write(rsp);
  endfunction
endclass
