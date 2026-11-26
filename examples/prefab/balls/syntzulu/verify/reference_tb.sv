module reference_tb;
bit clock, reset=1, wr, feed, rd;
bit [6:0] addr;
bit [7:0] out_addr;
bit [63:0] weights;
bit [3:0] group_bits;
wire done; wire [31:0] result;
always #5 clock=~clock;
ReferenceCore dut(.*);
localparam logic [31:0] expected [0:31] = '{
  32'h00280068, 32'h005000d0, 32'h00780138, 32'h00a001a0,
  32'h00c80208, 32'h00f00270, 32'h011802d8, 32'h01400340,
  32'h0023005d, 32'h004700ba, 32'h006b0117, 32'h008f0174,
  32'h00b301d2, 32'h00d7022f, 32'h00fb028c, 32'h011e02e9,
  32'h004700bb, 32'h008f0176, 32'h00d70232, 32'h012002ed,
  32'h016803a9, 32'h01b00000, 32'h01f90000, 32'h02400000,
  32'h0067010f, 32'h00d0021f, 32'h0138032f, 32'h01a20000,
  32'h020a0000, 32'h02730270, 32'h02dc02d8, 32'h03440340
};
initial begin
repeat(8) @(negedge clock); reset=0;
for(int i=0;i<128;i++) begin
  wr=1; addr=i; weights=i<96 ? 64'h0102040810203040 : 64'h0102030405060708 * (1+((i>>2)&7));
  @(negedge clock);
end
wr=0;
for (int frame=0;frame<4;frame++) begin
  bit [15:0] spikes;
  case (frame)
    0: spikes=1;
    1: spikes=0;
    2: spikes=16'h8000;
    3: spikes=16'hffff;
  endcase
  rd=0;
  for(int i=0;i<4;i++) begin feed=1; group_bits=spikes[4*i+:4]; @(negedge clock); end
  feed=0;
  wait(done); @(negedge clock); @(negedge clock);
  for(int i=0;i<8;i++) begin rd=1; out_addr=i; @(negedge clock);
    if (result !== expected[8*frame+i]) $fatal(1,"frame %0d row %0d: %h",frame,i,result);
    $display("FRAME %0d ROW %0d %08x",frame,i,result);
  end
end
$finish;
end
initial begin #100000; $fatal(1,"timeout");end
endmodule
