class stimulus extends uvm_sequence #(request_item);
  `uvm_object_utils(stimulus)
  int unsigned sent = 0;
  function new(string name = "stimulus");
    super.new(name);
  endfunction
  task body();
    request_item req = request_item::type_id::create("request");
    int unsigned rounding;
    start_item(req);
    fpu_case(fpu_cancel_case(), req.instruction, req.integer_value, req.a, req.rhs, req.c,
             rounding);
    req.rounding = rounding[2:0];
    req.stall = 0;
    req.cancel = 1;
    finish_item(req);
    for (int i = 0; i < fpu_case_count(); i++) begin
      start_item(req);
      fpu_case(i, req.instruction, req.integer_value, req.a, req.rhs, req.c, rounding);
      req.rounding = rounding[2:0];
      req.stall = (i % 2) * 3;
      req.cancel = 0;
      finish_item(req);
      sent++;
    end
  endtask
endclass
