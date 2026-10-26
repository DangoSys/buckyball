class stimulus extends uvm_sequence #(request_item);
  `uvm_object_utils(stimulus)
  int unsigned sent = 0;
  reset_controller resetter;
  function new(string name = "stimulus");
    super.new(name);
  endfunction
  task body();
    request_item req = request_item::type_id::create("request");
    logic [63:0] operands[8] = '{0, 1, 31, 32, 63, '1, 64'h8000000000000000, 64'h55555555aaaaaaaa};
    int operations[24] = '{
        0,
        10,
        1,
        5,
        11,
        4,
        6,
        7,
        8,
        9,
        12,
        13,
        14,
        15,
        17,
        18,
        19,
        24,
        25,
        26,
        28,
        29,
        30,
        31
    };
    int unary[8] = '{0, 1, 2, 'h287, 'h6b8, 'h080, 'h604, 'h605};
    for (int word = 0; word < 2; word++) begin
      foreach (operations[op]) begin
        if (word && !(operations[op] inside {0, 10, 1, 5, 11, 17, 18})) continue;
        foreach (operands[a])
        foreach (operands[b]) begin
          start_item(req);
          req.a   = operands[a];
          req.rhs = operands[b];
          req.fn  = operations[op];
          req.dw  = !word;
          finish_item(req);
          sent++;
        end
      end
      foreach (unary[op]) begin
        if (word && op >= 3) continue;
        foreach (operands[a]) begin
          start_item(req);
          req.a   = operands[a];
          req.rhs = unary[op];
          req.fn  = 16;
          req.dw  = !word;
          finish_item(req);
          sent++;
        end
      end
    end
  endtask
endclass
