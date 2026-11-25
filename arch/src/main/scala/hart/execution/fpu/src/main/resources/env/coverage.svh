`uvm_analysis_imp_decl(_request)
`uvm_analysis_imp_decl(_response)
class coverage extends uvm_component;
  `uvm_component_utils(coverage)
  uvm_analysis_imp_request #(request_item, coverage) requests;
  uvm_analysis_imp_response #(response_item, coverage) responses;
  request_item active;
  covergroup behavior with function sample (request_item req, response_item rsp);
    option.per_instance = 1;
    opcode: coverpoint req.instruction[6:0] {
      bins arithmetic = {7'h53}; bins fused[] = {7'h43, 7'h47, 7'h4b, 7'h4f}; bins invalid = {0};
    }
    operation: coverpoint req.instruction[31:27] iff (req.instruction[6:0] == 7'h53) {
      bins supported[] = {0, 1, 2, 3, 4, 5, 8, 11, 20, 24, 26, 28, 30}; bins invalid = {31};
    }
    format: coverpoint req.instruction[26:25] {bins formats[] = {0, 1}; bins invalid = {[2 : 3]};}
    rounding: coverpoint (req.instruction[14:12] == 7 ? req.rounding : req.instruction[14:12])
      iff (req.instruction[6:0] inside {7'h43,7'h47,7'h4b,7'h4f} ||
           (req.instruction[6:0] == 7'h53 && req.instruction[31:27] inside {0,1,2,3,8,11,24,26})) {
      bins modes[] = {[0 : 4]}; bins invalid = {[5 : 7]};
    }
    integer_type: coverpoint req.instruction[21:20] iff (req.instruction[31:27] inside {24, 26}) {
      bins types[] = {[0 : 3]};
    }
    flags: coverpoint rsp.flags {bins exceptions[] = {0, 1, 3, 5, 8, 16};}
    result_kind: coverpoint rsp.integer_result;
    illegal: coverpoint rsp.illegal;
    classification: coverpoint rsp.value[9:0]
      iff (req.instruction[6:0] == 7'h53 && req.instruction[31:27] == 28 &&
           req.instruction[14:12] == 1 && !rsp.illegal) {
      bins classes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    }
    backpressure: coverpoint req.stall {bins stalls[] = {0, 3};}
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    requests  = new("requests", this);
    responses = new("responses", this);
    behavior  = new();
  endfunction
  function void write_request(request_item req);
    active = req;
  endfunction
  function void write_response(response_item rsp);
    behavior.sample(active, rsp);
  endfunction
endclass
