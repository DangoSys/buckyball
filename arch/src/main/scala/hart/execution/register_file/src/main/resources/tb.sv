module register_file_tb;
  import uvm_pkg::*;
  import register_file_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  clock_if timing (clock);
  port_if bus (
      clock,
      timing.reset
  );
  RegisterFile dut (
      .clock(clock),
      .reset(bus.reset),
      .io_readAddress_0(bus.read0),
      .io_readAddress_1(bus.read1),
      .io_readData_0(bus.data0),
      .io_readData_1(bus.data1),
      .io_write_valid(bus.sample && bus.write),
      .io_write_bits_address(bus.address),
      .io_write_bits_data(bus.data)
  );
  initial begin
    uvm_config_db#(virtual clock_if)::set(null, "uvm_test_top*", "timing", timing);
    uvm_config_db#(virtual port_if)::set(null, "uvm_test_top*", "vif", bus);
    run_test("protocol_test");
  end
endmodule
