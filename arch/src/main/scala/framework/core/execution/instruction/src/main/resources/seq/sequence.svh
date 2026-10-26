class stimulus extends uvm_sequence #(request_item);
  `uvm_object_utils(stimulus)
  int unsigned sent = 0;
  function new(string name = "stimulus");
    super.new(name);
  endfunction
  task body();
    request_item req = request_item::type_id::create("request");
    int funct7s[15] = '{0, 1, 4, 8, 9, 12, 16, 32, 33, 48, 64, 80, 96, 112, 127};
    int system_words[6] = '{'h00000073, 'h00100073, 'h30200073, 'h10200073, 'h10500073, 'h12000073};
    for (int bits = 0; bits < 65536; bits++) begin
      if ((bits & 3) == 3) continue;
      start_item(req);
      req.bits = bits;
      finish_item(req);
      sent++;
      if ((bits & 'hfc) == 0) begin
        start_item(req);
        req.bits = 'hffff0000 | bits;
        finish_item(req);
        sent++;
      end
    end
    for (int opcode = 3; opcode < 128; opcode += 4)
      for (int funct3 = 0; funct3 < 8; funct3++)
        foreach (funct7s[index])
          for (int rd = 0; rd < 32; rd++) begin
            start_item(req);
            req.bits=(funct7s[index]<<25)|(((rd+13)&31)<<20)|(((rd+7)&31)<<15)|(funct3<<12)|(rd<<7)|opcode;
            finish_item(req);
            sent++;
          end
    foreach (system_words[index]) begin
      start_item(req);
      req.bits = system_words[index];
      finish_item(req);
      sent++;
    end
  endtask
endclass
