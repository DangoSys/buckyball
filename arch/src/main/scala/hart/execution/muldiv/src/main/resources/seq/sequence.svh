class stimulus extends uvm_sequence #(request_item);
  `uvm_object_utils(stimulus)
  int unsigned sent = 0;
  function new(string name = "stimulus");
    super.new(name);
  endfunction
  task body();
    request_item req = request_item::type_id::create("request");
    logic [63:0] operands[10] = '{
        0,
        1,
        3,
        '1,
        64'h8000000000000000,
        64'h7fffffffffffffff,
        64'h80000000,
        64'h7fffffff,
        64'hffffffff,
        64'h123456789abcdef0
    };
    start_item(req);
    req.a = 123456789;
    req.rhs = 7;
    req.operation = 4;
    req.word = 0;
    req.stall = 0;
    req.cancel = 1;
    finish_item(req);
    for (int word = 0; word < 2; word++)
      for (int operation = 0; operation < 8; operation++) begin
        if (word && operation inside {1, 2, 3}) continue;
        foreach (operands[a])
        foreach (operands[b]) begin
          start_item(req);
          req.a = operands[a];
          req.rhs = operands[b];
          req.operation = operation;
          req.word = word;
          req.stall = ((a + b) % 2) * 3;
          req.cancel = 0;
          finish_item(req);
          sent++;
        end
      end
  endtask
endclass
