class model extends reference_model #(request_item, response_item);
  `uvm_component_utils(model)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void write(request_item req);
    response_item rsp = response_item::type_id::create("expected");
    int unsigned expanded, atomic, function_code;
    longint unsigned immediate;
    byte unsigned illegal, word, immediate_mode;
    instruction_expected(req.bits, expanded, illegal, immediate, atomic, function_code, word,
                         immediate_mode);
    rsp.set_name($sformatf("expected_%08h", req.bits));
    rsp.compressed = req.bits[1:0] != 3;
    rsp.illegal = illegal[0];
    rsp.expanded = illegal ? 0 : expanded;
    rsp.opcode = illegal ? 0 : expanded[6:0];
    rsp.funct3 = illegal ? 0 : expanded[14:12];
    rsp.funct7 = illegal ? 0 : expanded[31:25];
    rsp.rs1 = illegal ? 0 : expanded[19:15];
    rsp.rs2 = illegal ? 0 : expanded[24:20];
    rsp.rd = illegal ? 0 : expanded[11:7];
    rsp.csr_address = illegal ? 0 : expanded[31:20];
    rsp.immediate = illegal ? 0 : immediate;
    rsp.atomic = illegal ? 0 : atomic[3:0];
    rsp.function_code = illegal ? 0 : function_code[4:0];
    rsp.word = illegal ? 0 : word[0];
    rsp.immediate_mode = illegal ? 0 : immediate_mode[0];
    expected_ap.write(rsp);
  endfunction
endclass
