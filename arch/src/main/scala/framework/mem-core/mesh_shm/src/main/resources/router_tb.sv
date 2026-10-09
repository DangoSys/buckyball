module router_tb;
  import uvm_pkg::*;
  import router_pkg::*;
  logic clock = 0;
  logic reset;
  always #5 clock = ~clock;
  axis_if #(128, 8, 4, 45) source_if[5] (clock);
  axis_if #(128, 8, 4, 45) sink_if[5] (clock);
  assign reset = source_if[0].reset;
  MeshRouter dut (
      .clock(clock),
      .reset(reset),
      .io_in_0_valid(source_if[0].tvalid),
      .io_in_0_ready(source_if[0].tready),
      .io_in_0_bits_tdata(source_if[0].tdata),
      .io_in_0_bits_tkeep(source_if[0].tkeep),
      .io_in_0_bits_tlast(source_if[0].tlast),
      .io_in_0_bits_tid(source_if[0].tid),
      .io_in_0_bits_tdest(source_if[0].tdest),
      .io_in_0_bits_tuser(source_if[0].tuser),
      .io_in_1_valid(source_if[1].tvalid),
      .io_in_1_ready(source_if[1].tready),
      .io_in_1_bits_tdata(source_if[1].tdata),
      .io_in_1_bits_tkeep(source_if[1].tkeep),
      .io_in_1_bits_tlast(source_if[1].tlast),
      .io_in_1_bits_tid(source_if[1].tid),
      .io_in_1_bits_tdest(source_if[1].tdest),
      .io_in_1_bits_tuser(source_if[1].tuser),
      .io_in_2_valid(source_if[2].tvalid),
      .io_in_2_ready(source_if[2].tready),
      .io_in_2_bits_tdata(source_if[2].tdata),
      .io_in_2_bits_tkeep(source_if[2].tkeep),
      .io_in_2_bits_tlast(source_if[2].tlast),
      .io_in_2_bits_tid(source_if[2].tid),
      .io_in_2_bits_tdest(source_if[2].tdest),
      .io_in_2_bits_tuser(source_if[2].tuser),
      .io_in_3_valid(source_if[3].tvalid),
      .io_in_3_ready(source_if[3].tready),
      .io_in_3_bits_tdata(source_if[3].tdata),
      .io_in_3_bits_tkeep(source_if[3].tkeep),
      .io_in_3_bits_tlast(source_if[3].tlast),
      .io_in_3_bits_tid(source_if[3].tid),
      .io_in_3_bits_tdest(source_if[3].tdest),
      .io_in_3_bits_tuser(source_if[3].tuser),
      .io_in_4_valid(source_if[4].tvalid),
      .io_in_4_ready(source_if[4].tready),
      .io_in_4_bits_tdata(source_if[4].tdata),
      .io_in_4_bits_tkeep(source_if[4].tkeep),
      .io_in_4_bits_tlast(source_if[4].tlast),
      .io_in_4_bits_tid(source_if[4].tid),
      .io_in_4_bits_tdest(source_if[4].tdest),
      .io_in_4_bits_tuser(source_if[4].tuser),
      .io_out_0_valid(sink_if[0].tvalid),
      .io_out_0_ready(sink_if[0].tready),
      .io_out_0_bits_tdata(sink_if[0].tdata),
      .io_out_0_bits_tkeep(sink_if[0].tkeep),
      .io_out_0_bits_tlast(sink_if[0].tlast),
      .io_out_0_bits_tid(sink_if[0].tid),
      .io_out_0_bits_tdest(sink_if[0].tdest),
      .io_out_0_bits_tuser(sink_if[0].tuser),
      .io_out_1_valid(sink_if[1].tvalid),
      .io_out_1_ready(sink_if[1].tready),
      .io_out_1_bits_tdata(sink_if[1].tdata),
      .io_out_1_bits_tkeep(sink_if[1].tkeep),
      .io_out_1_bits_tlast(sink_if[1].tlast),
      .io_out_1_bits_tid(sink_if[1].tid),
      .io_out_1_bits_tdest(sink_if[1].tdest),
      .io_out_1_bits_tuser(sink_if[1].tuser),
      .io_out_2_valid(sink_if[2].tvalid),
      .io_out_2_ready(sink_if[2].tready),
      .io_out_2_bits_tdata(sink_if[2].tdata),
      .io_out_2_bits_tkeep(sink_if[2].tkeep),
      .io_out_2_bits_tlast(sink_if[2].tlast),
      .io_out_2_bits_tid(sink_if[2].tid),
      .io_out_2_bits_tdest(sink_if[2].tdest),
      .io_out_2_bits_tuser(sink_if[2].tuser),
      .io_out_3_valid(sink_if[3].tvalid),
      .io_out_3_ready(sink_if[3].tready),
      .io_out_3_bits_tdata(sink_if[3].tdata),
      .io_out_3_bits_tkeep(sink_if[3].tkeep),
      .io_out_3_bits_tlast(sink_if[3].tlast),
      .io_out_3_bits_tid(sink_if[3].tid),
      .io_out_3_bits_tdest(sink_if[3].tdest),
      .io_out_3_bits_tuser(sink_if[3].tuser),
      .io_out_4_valid(sink_if[4].tvalid),
      .io_out_4_ready(sink_if[4].tready),
      .io_out_4_bits_tdata(sink_if[4].tdata),
      .io_out_4_bits_tkeep(sink_if[4].tkeep),
      .io_out_4_bits_tlast(sink_if[4].tlast),
      .io_out_4_bits_tid(sink_if[4].tid),
      .io_out_4_bits_tdest(sink_if[4].tdest),
      .io_out_4_bits_tuser(sink_if[4].tuser)
  );
  for (genvar i = 0; i < 5; i++) begin
    initial begin
      uvm_config_db#(virtual axis_if #(128, 8, 4, 45))::set(null, "uvm_test_top*", $sformatf(
                                                            "source%0d", i), source_if[i]);
      uvm_config_db#(virtual axis_if #(128, 8, 4, 45))::set(null, "uvm_test_top*", $sformatf(
                                                            "sink%0d", i), sink_if[i]);
    end
  end
  initial run_test("protocol_test");
endmodule
