class access_sequence extends uvm_sequence #(request_item);
  `uvm_object_utils(access_sequence)
  int unsigned expected_responses = 0;
  int unsigned expected_cancellations = 0;

  function new(string name = "access_sequence");
    super.new(name);
  endfunction

  task access (bit wr, bit [`BANK_ROW_BITS-1:0] addr, bit [31:0] data, bit [3:0] mask,
               bit [`BANK_TAG_BITS-1:0] tag, int unsigned stall, bit cancel_response = 0);
    request_item req = request_item::type_id::create("access");
    start_item(req);
    req.write = wr;
    req.addr = addr;
    req.data = data;
    req.mask = mask;
    req.tag = tag;
    req.stall_cycles = stall;
    req.cancel_response = cancel_response;
    finish_item(req);
    if (cancel_response) expected_cancellations++;
    else expected_responses++;
  endtask

  task body();
    request_item req = request_item::type_id::create("reset");
    start_item(req);
    req.reset_command = 1;
    finish_item(req);
    for (int a = 0; a < `BANK_ROWS; a++) access (1, a, 32'h9e3779b9 * (a + 1), 15, a, 0);
    for (int a = 0; a < `BANK_ROWS; a++) access (0, a, 0, 0, 255 - a, 2);
    for (int m = 0; m < 16; m++) begin
      access (1, m % `BANK_ROWS, ~(32'h9e3779b9 * (m + 1)), m, m, 5);
      access (0, m % `BANK_ROWS, 0, 0, m, 0);
    end
    access (1, `BANK_ROWS - 1, 'hface1234, 15, 255, 0, 1);
    access (0, `BANK_ROWS - 1, 0, 0, 42, 3);
    access (0, `BANK_ROWS - 1, 0, 15, 255, 0, 1);
    access (0, `BANK_ROWS - 1, 0, 0, 42, 3);
    for (int a = 0; a < `BANK_ROWS; a++) access (0, a, 0, 0, a, 0);
  endtask
endclass
