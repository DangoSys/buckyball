module MagiaClockGate (
    input  logic clk_i,
    input  logic en_i,
    input  logic test_en_i,
    output logic clk_o
);
  logic clk_en;

  // Sample synchronous enables while the generated clock is low.
  always_ff @(negedge clk_i) clk_en <= en_i | test_en_i;

  assign clk_o = clk_i & clk_en;
endmodule
