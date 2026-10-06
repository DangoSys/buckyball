module bad_node_tb;
  import uvm_pkg::*;
  import chi_bad_node_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  chi_codec_if vif (clock);
  ChiMeshCodecLoopback dut (
      .clock(clock),
      .reset(vif.reset),
      .io_pause(vif.pause),
      .io_in_valid(vif.valid),
      .io_in_ready(vif.ready),
      .io_in_bits_targetNode(vif.target),
      .io_in_bits_flit(vif.data),
      .io_out_valid(vif.out_valid),
      .io_out_ready(vif.out_ready),
      .io_out_bits(vif.out_data),
      .io_observedFire(vif.chunk_fire),
      .io_observed_head(vif.chunk_head),
      .io_observed_tail(vif.chunk_tail),
      .io_observed_dstX(vif.chunk_x),
      .io_observed_dstY(vif.chunk_y)
  );
  initial begin
    uvm_config_db#(virtual chi_codec_if)::set(null, "uvm_test_top", "vif", vif);
    run_test("protocol_test");
  end
endmodule
