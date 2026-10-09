interface ant_control_if (
    input logic clock,
    reset
);
  import uvm_pkg::*;
  `include "uvm_macros.svh"
  logic valid = 0, ready, reply_valid, reply_ready = 0;
  logic [6:0] operation = 0;
  logic [31:0] context_id = 0, field = 0;
  logic [63:0] data = 0, reply;
  int checked = 0;
  task automatic transfer(int op, int ctx, int f, longint unsigned value,
                          output longint unsigned result);
    logic [63:0] held;
    @(negedge clock);
    operation = op;
    context_id = ctx;
    field = f;
    data = value;
    valid = 1;
    do @(posedge clock); while (!ready);
    @(negedge clock);
    valid = 0;
    wait (reply_valid);
    held = reply;
    repeat (3) begin
      @(negedge clock);
      if (!reply_valid || reply !== held)
        `uvm_fatal("CONTROL", $sformatf(
                   "op%0d ctx%0d field%0d: reply changed from %h to %h", op, ctx, f, held, reply))
    end
    result = reply;
    reply_ready = 1;
    @(negedge clock);
    reply_ready = 0;
    checked++;
  endtask
endinterface
