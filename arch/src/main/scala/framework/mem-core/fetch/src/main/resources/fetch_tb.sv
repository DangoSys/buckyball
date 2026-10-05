`include "fetch_config.svh"
`ifndef FETCH_TOP
`define FETCH_TOP fetch_tb
`endif
module `FETCH_TOP;
  import uvm_pkg::*;
  import fetch_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  fetch_control_if control (clock);
  stream_if #(`FETCH_REQUEST_WIDTH) request (
      clock,
      control.reset
  );
  stream_if #(`FETCH_RESPONSE_WIDTH) response (
      clock,
      control.reset
  );
  // CPU redirection explicitly cancels a held frontend packet; it never cancels the word port.
  stream_if #(`FETCH_PACKET_WIDTH) sink_if (
      clock,
      control.reset || control.redirect_valid || control.flush
  );
  stream_if #(1) maintenance (
      clock,
      control.reset
  );
  stream_if #(1) maintained (
      clock,
      control.reset
  );
  bit read_pending = 0, maintenance_pending = 0;
  always @(posedge clock) begin
    if (control.reset) begin
      read_pending = 0;
      maintenance_pending = 0;
    end else begin
      if (request.valid && request.ready) begin
        if (read_pending || maintenance_pending) $fatal(1, "Fetch overlapped backend transactions");
        read_pending = 1;
      end
      if (response.valid && response.ready) begin
        if (!read_pending) $fatal(1, "Fetch accepted an unsolicited word response");
        read_pending = 0;
      end
      if (maintenance.valid && maintenance.ready) begin
        if (read_pending || maintenance_pending)
          $fatal(1, "Maintenance began before draining the word");
        maintenance_pending = 1;
      end
      if (maintained.valid && maintained.ready) begin
        if (!maintenance_pending)
          $fatal(1, "Fetch accepted an unsolicited maintenance acknowledgement");
        maintenance_pending = 0;
      end
      if ((control.redirect_valid || control.flush) && sink_if.valid)
        $fatal(1, "Cancelled packet was visible to the CPU");
    end
  end
  Fetch dut (
      .clock(clock),
      .reset(control.reset),
      .io_resetVector(control.reset_vector),
      .io_context_privilege(control.privilege),
      .io_context_satp(control.satp),
      .io_context_sum(control.sum),
      .io_context_mxr(control.mxr),
      .io_redirect_valid(control.redirect_valid),
      .io_redirect_bits(control.redirect_pc),
      .io_flush(control.flush),
      .io_invalidate(1'b0),
      .io_npc(control.npc),
      `include "fetch_ports.svh"
  );
  initial begin
    uvm_config_db#(virtual fetch_control_if)::set(null, "uvm_test_top*", "control", control);
    uvm_config_db#(virtual stream_if #(`FETCH_REQUEST_WIDTH))::set(null, "uvm_test_top*", "request",
                                                                   request);
    uvm_config_db#(virtual stream_if #(`FETCH_RESPONSE_WIDTH))::set(null, "uvm_test_top*",
                                                                    "response", response);
    uvm_config_db#(virtual stream_if #(`FETCH_PACKET_WIDTH))::set(null, "uvm_test_top*", "packet",
                                                                  sink_if);
    uvm_config_db#(virtual stream_if #(1))::set(null, "uvm_test_top*", "maintenance", maintenance);
    uvm_config_db#(virtual stream_if #(1))::set(null, "uvm_test_top*", "maintained", maintained);
    run_test("protocol_test");
  end
endmodule
