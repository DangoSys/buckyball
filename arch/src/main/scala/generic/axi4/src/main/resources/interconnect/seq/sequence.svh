class traffic extends uvm_sequence #(batch);
  `uvm_object_utils(traffic)
  function new(string name = "traffic");
    super.new(name);
  endfunction
  task body();
    batch req;
    for (int n = 0; n < 16; n++) begin
      req = batch::type_id::create("batch");
      start_item(req);
      req.serial = n;
      case (n % 4)
        0: req.beats = 1;
        1: req.beats = 2;
        2: req.beats = 16;
        3: req.beats = 256;
      endcase
      req.stalls = n >= 8;
      finish_item(req);
    end
  endtask
endclass
