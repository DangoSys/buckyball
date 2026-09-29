interface bankset_if (
    input logic clock
);
  logic reset;
  logic valid;
  logic ready;
  logic write;
  logic [7:0] addr;
  logic [31:0] beats;
  logic done_valid;
  logic done_ready;
  logic error;

  assert property (@(posedge clock) disable iff (reset)
    done_valid && !done_ready |=> done_valid && $stable(
      error
  ))
  else $fatal(1, "BankSet completion changed under backpressure");
endinterface
