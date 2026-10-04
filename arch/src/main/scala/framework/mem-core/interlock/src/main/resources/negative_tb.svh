module `INTERLOCK_NEGATIVE_TB;
  import uvm_pkg::*;
  import interlock_pkg::*;
  `include "uvm_macros.svh"
  `include "interlock_config.svh"
  `define ILF(B, K, F) B[`INTERLOCK_``K``_``F``_OFFSET +: `INTERLOCK_``K``_``F``_WIDTH]
  logic clock = 0, reset = 1, cpu_allow;
  always #5 clock = ~clock;
  interlock_control_if ctl (clock);
  stream_if #(
      .WIDTH(`INTERLOCK_DISPATCH_WIDTH)
  ) dispatch (
      clock,
      reset
  );
  stream_if #(
      .WIDTH(`INTERLOCK_CANCEL_WIDTH)
  ) cancel (
      clock,
      reset
  );
  stream_if #(
      .WIDTH(`INTERLOCK_ACCESS_INFO_WIDTH)
  ) access_info (
      clock,
      reset
  );
  stream_if #(
      .WIDTH(`INTERLOCK_MAINTENANCE_WIDTH)
  ) maintenance (
      clock,
      reset
  );
  stream_if #(
      .WIDTH(`INTERLOCK_MAINTAINED_WIDTH)
  ) maintained (
      clock,
      reset
  );
  stream_if #(
      .WIDTH(`INTERLOCK_GRANT_WIDTH)
  ) grant (
      clock,
      reset
  );
  stream_if #(
      .WIDTH(`INTERLOCK_DONE_WIDTH)
  ) done (
      clock,
      reset
  );
  stream_if #(
      .WIDTH(`INTERLOCK_COMPLETE_WIDTH)
  ) complete (
      clock,
      reset
  );
  stream_if #(
      .WIDTH(`INTERLOCK_CPU_QUERY_WIDTH)
  ) cpu_query (
      clock,
      reset
  );
  Interlock dut (
      .clock(clock),
      .reset(reset),
      `include "interlock_ports.svh"
  );
  task automatic reserve(int id);
    @(negedge clock);
    dispatch.bits = '0;
    `ILF(dispatch.bits, DISPATCH, ID) = id;
    dispatch.valid = 1;
    do @(posedge clock); while (!dispatch.ready);
    @(negedge clock);
    dispatch.valid = 0;
  endtask
  task automatic info(int id, bit memory, longint unsigned base, bytes, bit write, bit last = 1);
    @(negedge clock);
    access_info.bits = '0;
    `ILF(access_info.bits, ACCESS_INFO, ID) = id;
    `ILF(access_info.bits, ACCESS_INFO, HASMEMORY) = memory;
    `ILF(access_info.bits, ACCESS_INFO, BASE) = base;
    `ILF(access_info.bits, ACCESS_INFO, BYTES) = bytes;
    `ILF(access_info.bits, ACCESS_INFO, WRITE) = write;
    `ILF(access_info.bits, ACCESS_INFO, LAST) = last;
    access_info.valid = 1;
    do @(posedge clock); while (!access_info.ready);
    @(negedge clock);
    access_info.valid = 0;
  endtask
  task automatic maintenance_accept();
    wait (maintenance.valid);
    @(negedge clock);
    maintenance.ready = 1;
    @(posedge clock);
    @(negedge clock);
    maintenance.ready = 0;
  endtask
  task automatic maintenance_ack(int id, bit ok);
    @(negedge clock);
    maintained.bits = '0;
    `ILF(maintained.bits, MAINTAINED, TAG) = id;
    `ILF(maintained.bits, MAINTAINED, OK) = ok;
    maintained.valid = 1;
  endtask
  initial begin
    uvm_config_db#(virtual interlock_control_if)::set(null, "*", "vif", ctl);
    run_test("protocol_test");
  end
  initial begin
    dispatch.valid = 0;
    dispatch.bits = '0;
    cancel.valid = 0;
    cancel.bits = '0;
    access_info.valid = 0;
    access_info.bits = '0;
    maintenance.ready = 0;
    maintained.valid = 0;
    maintained.bits = '0;
    grant.ready = 0;
    done.valid = 0;
    done.bits = '0;
    complete.ready = 0;
    cpu_query.bits = '0;
    cpu_query.valid = 0;
    cpu_query.ready = 0;
    wait (ctl.start);
    repeat (4) @(posedge clock);
    @(negedge clock);
    reset = 0;
    `include `INTERLOCK_NEGATIVE_SCENARIO
    repeat (20) @(posedge clock);
    `uvm_fatal("MISSING_ASSERTION", "illegal input did not produce the declared RTL assertion")
  end
endmodule
