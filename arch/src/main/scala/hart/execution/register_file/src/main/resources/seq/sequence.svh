class stimulus extends uvm_sequence #(request_item);
  `uvm_object_utils(stimulus)
  int unsigned sent = 0;
  reset_controller resetter;
  function new(string name = "stimulus");
    super.new(name);
  endfunction
  task body();
    request_item req = request_item::type_id::create("request");
    for (int i = 1; i < 32; i++) begin
      start_item(req);
      req.read0 = 0;
      req.read1 = 0;
      req.write = 1;
      req.address = i;
      req.data = 64'h0102040810204081 * i;
      finish_item(req);
      sent++;
    end
    resetter.pulse(2);
    for (int i = 0; i < 32; i++)
      for (int j = 0; j < 32; j++) begin
        start_item(req);
        req.read0 = i;
        req.read1 = j;
        req.write = 0;
        req.address = 0;
        req.data = 0;
        finish_item(req);
        sent++;
      end
    for (int i = 0; i < 32; i++) begin
      start_item(req);
      req.read0 = i;
      req.read1 = i;
      req.write = 1;
      req.address = i;
      req.data = '1;
      finish_item(req);
      sent++;
      start_item(req);
      req.write = 0;
      finish_item(req);
      sent++;
    end
  endtask
endclass
