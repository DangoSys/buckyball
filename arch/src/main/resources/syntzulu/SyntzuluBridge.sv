module SyntzuluBridge(
  input wire clock, reset, cmd_valid, output wire cmd_ready,
  input wire [1:0] cmd_op, input wire [39:0] cmd_data,
  output wire rsp_valid, input wire rsp_ready, output reg [31:0] rsp_data
);
  localparam IDLE=0, INPUT_GROUPS=1, COMPUTE=2, CAPTURE=3, RESPONSE=4;
  reg [2:0] state;
  reg [15:0] frame;
  reg [1:0] group_index;
  wire feed = state == INPUT_GROUPS;
  wire [3:0] group_bits = frame[3:0];
  assign cmd_ready = state == IDLE;
  assign rsp_valid = state == RESPONSE;
  wire wr = cmd_ready && cmd_valid && cmd_op == 0;
  wire rd = cmd_ready && cmd_valid && cmd_op == 2;
  wire [63:0] weights = {2{cmd_data[31:0]}};
  wire [6:0] addr = cmd_data[38:32];
  wire [7:0] out_addr = {5'd0,cmd_data[2:0]};
  wire done;
  wire [31:0] result;
  snn_lp #(
    .WIDTH(16),
    .MAX_NEURONS(16),
    .MAX_SYNAPSES(16),
    .WEIGHT_DEPTH_12(128),
    .WEIGHT_DEPTH_34(128),
    .INPUT_SPIKE_1(16),
    .NEURON_1(16),
    .WEIGHTS_FILE_1(""),
    .threshold_1(1),
    .INPUT_SPIKE_2(16),
    .NEURON_2(16),
    .WEIGHTS_FILE_2(""),
    .threshold_2(1),
    .INPUT_SPIKE_3(16),
    .NEURON_3(16),
    .WEIGHTS_FILE_3(""),
    .threshold_3(1),
    .INPUT_SPIKE_4(16),
    .NEURON_4(16),
    .WEIGHTS_FILE_4(""),
    .threshold_4(1024)
  ) core(.clk(clock), .rst(reset), .en(feed), .active_group_in(|group_bits), .spike_in(group_bits), .valid(done), .valid_spike(), .spike_out(), .integrated_neuron(),
    .weight_mem_L1_wren({7'd0, wr && cmd_data[39] == 1'b0}),
    .weight_mem_L1_wr_addr(addr),
    .weight_mem_L1_ena(wr),
    .weight_mem_L1_data_in(weights[0 +: 16]),
    .weight_mem_L1_data_out(),
    .weight_mem_L2_wren({7'd0, wr && cmd_data[39] == 1'b0}),
    .weight_mem_L2_wr_addr(addr),
    .weight_mem_L2_ena(wr),
    .weight_mem_L2_data_in(weights[16 +: 16]),
    .weight_mem_L2_data_out(),
    .weight_mem_L3_wren({7'd0, wr && cmd_data[39] == 1'b1}),
    .weight_mem_L3_wr_addr(addr),
    .weight_mem_L3_ena(wr),
    .weight_mem_L3_data_in(weights[32 +: 16]),
    .weight_mem_L3_data_out(),
    .weight_mem_L4_wren({7'd0, wr && cmd_data[39] == 1'b1}),
    .weight_mem_L4_wr_addr(addr),
    .weight_mem_L4_ena(wr),
    .weight_mem_L4_data_in(weights[48 +: 16]),
    .weight_mem_L4_data_out(),
    .i_spike_mem_adr('0), .i_spike_mem_rd_en('0), .i_spike_mem_wr_en('0), .i_spike_mem_dat('0), .o_spike_mem_dat(),
    .snn_input_channels('0), .neuron_1('0), .neuron_2('0), .neuron_3('0), .neuron_4('0), .layers('0),
    .output_buffer_ren(rd), .output_buffer_addr(out_addr), .output_buffer_out(result));

  always @(posedge clock) begin
    if (reset) begin
      state <= IDLE; frame <= 0; group_index <= 0; rsp_data <= 0;
    end else case (state)
      IDLE: if (cmd_valid) begin
        rsp_data <= 0;
        case (cmd_op)
          0: state <= RESPONSE;
          1: begin frame <= cmd_data[15:0]; group_index <= 0; state <= INPUT_GROUPS; end
          2: state <= CAPTURE;
          default: state <= RESPONSE;
        endcase
      end
      INPUT_GROUPS: begin
        frame <= frame >> 4;
        group_index <= group_index + 1'b1;
        if (group_index == 3) state <= COMPUTE;
      end
      COMPUTE: if (done) state <= RESPONSE;
      CAPTURE: begin rsp_data <= result; state <= RESPONSE; end
      RESPONSE: if (rsp_ready) state <= IDLE;
    endcase
  end
endmodule
