module tc_sram #(
    parameter int unsigned NumWords = 1024,
    parameter int unsigned DataWidth = 128,
    parameter int unsigned ByteWidth = 8,
    parameter int unsigned NumPorts = 2,
    parameter int unsigned Latency = 1,
    parameter SimInit = "none",
    parameter bit PrintSimCfg = 0,
    parameter ImplKey = "none",
    parameter FPGAImplKey = "auto",
    parameter int unsigned AddrWidth = (NumWords > 1) ? $clog2(NumWords) : 1,
    parameter int unsigned BeWidth = (DataWidth + ByteWidth - 1) / ByteWidth,
    parameter type addr_t = logic [AddrWidth-1:0],
    parameter type data_t = logic [DataWidth-1:0],
    parameter type be_t = logic [BeWidth-1:0]
) (
    input logic clk_i,
    input logic rst_ni,
    input logic [NumPorts-1:0] req_i,
    input logic [NumPorts-1:0] we_i,
    input addr_t [NumPorts-1:0] addr_i,
    input data_t [NumPorts-1:0] wdata_i,
    input be_t [NumPorts-1:0] be_i,
    output data_t [NumPorts-1:0] rdata_o
);
  // Reset controls access state; software maintains SRAM contents.
  (* ram_style = FPGAImplKey *) data_t memory[NumWords];
  addr_t [NumPorts-1:0] read_address;

  always_ff @(posedge clk_i) begin
    for (int port = 0; port < NumPorts; port++) begin
      if (rst_ni && req_i[port] && we_i[port]) begin
        for (int bit_index = 0; bit_index < DataWidth; bit_index++) begin
          if (be_i[port][bit_index/ByteWidth])
            memory[addr_i[port]][bit_index] <= wdata_i[port][bit_index];
        end
      end
    end
  end

  for (genvar port = 0; port < NumPorts; port++) begin : ports
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) read_address[port] <= '0;
      else if (req_i[port] && !we_i[port]) read_address[port] <= addr_i[port];
    end
    wire [DataWidth-1:0] read_word = memory[
        (req_i[port] && !we_i[port]) ? addr_i[port] : read_address[port]];
    if (Latency == 0) begin : asynchronous_read
      assign rdata_o[port] = read_word;
    end else begin : synchronous_read
      data_t [Latency-1:0] pipeline;
      always_ff @(posedge clk_i) begin
        if (rst_ni) begin
          pipeline[Latency-1] <= read_word;
          for (int stage = 0; stage < Latency - 1; stage++) pipeline[stage] <= pipeline[stage+1];
        end
      end
      assign rdata_o[port] = pipeline[0];
    end
  end
endmodule
