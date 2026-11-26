`define BRAM
`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08.11.2022 11:18:47
// Design Name: 
// Module Name: BRAM_singlePort_readFirst
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module BRAM_singlePort_readFirst #(
  parameter RAM_WIDTH = 4,                  // Specify RAM data width
  parameter RAM_DEPTH = 64,                  // Specify RAM depth (number of entries)
  parameter RAM_PERFORMANCE = "HIGH_PERFORMANCE", // Select "HIGH_PERFORMANCE" or "LOW_LATENCY" 
  parameter INIT_FILE = ""                       // Specify name/location of RAM initialization file if using one (leave blank if not)
)
(
  input [clogb2(RAM_DEPTH-1)-1:0] addra,  // Port A address bus, width determined from RAM_DEPTH
  input [clogb2(RAM_DEPTH-1)-1:0] addrb,  // Port B address bus, width determined from RAM_DEPTH
  input [RAM_WIDTH-1:0] dina,           // Port A RAM input data
  input clk,                           // Clock
  input wea,                            // Port A write enable
  input ena,                            // Port A RAM Enable, for additional power savings, disable port when not in use
  input enb,                            // Port B RAM Enable, for additional power savings, disable port when not in use
  input rst,                           // Port A and B output reset (does not affect memory contents)
  input regceb,                         // Port B output register enable
  
  output [RAM_WIDTH-1:0] doutb                   // Port B RAM output data
    );

  reg [RAM_WIDTH-1:0] ram [RAM_DEPTH-1:0];
  reg [RAM_WIDTH-1:0] ram_data_b = {RAM_WIDTH{1'b0}};

  generate
	  genvar idx;
	  for(idx = 0; idx < 16; idx = idx+1) begin
		wire [RAM_WIDTH-1:0] tmp;
		assign tmp = ram[idx];
	  end
  endgenerate

  // The following code either initializes the memory values to a specified file or to all zeros to match hardware
  generate
    if (INIT_FILE != "") begin: use_init_file
      initial
        $readmemh(INIT_FILE, ram, 0, RAM_DEPTH-1);
    end else begin: init_bram_to_zero
      integer ram_index;
      initial
        for (ram_index = 0; ram_index < RAM_DEPTH; ram_index = ram_index + 1)
          ram[ram_index] = {RAM_WIDTH{1'b0}};
    end
  endgenerate

  always @(posedge clk)
    if (ena) begin
      if (wea)
        ram[addra] <= dina;
    end

  always @(posedge clk)
    if (enb) begin
      ram_data_b <= ram[addrb];
    end

  //  The following code generates HIGH_PERFORMANCE (use output register) or LOW_LATENCY (no output register)
  generate
    if (RAM_PERFORMANCE == "LOW_LATENCY") begin: no_output_register

      // The following is a 1 clock cycle read latency at the cost of a longer clock-to-out timing
       assign doutb = ram_data_b;

    end else begin: output_register

      // The following is a 2 clock cycle read latency with improve clock-to-out timing

      reg [RAM_WIDTH-1:0] doutb_reg = {RAM_WIDTH{1'b0}};

      always @(posedge clk)
        if (rst)
          doutb_reg <= {RAM_WIDTH{1'b0}};
        else if (regceb)
          doutb_reg <= ram_data_b;

      assign doutb = doutb_reg;

    end
  endgenerate

  //  The following function calculates the address width based on specified RAM depth
  function integer clogb2;
    input integer depth;
      for (clogb2=0; depth>0; clogb2=clogb2+1)
        depth = depth >> 1;
  endfunction
							
endmodule

module SPRAM_singlePort_readFirst #(
  parameter RAM_WIDTH = 4,                  // Specify RAM data width
  parameter RAM_DEPTH = 64,                  // Specify RAM depth (number of entries)
  parameter RAM_PERFORMANCE = "HIGH_PERFORMANCE", // Select "HIGH_PERFORMANCE" or "LOW_LATENCY" 
  parameter INIT_FILE = ""                       // Specify name/location of RAM initialization file if using one (leave blank if not)
)
(
  input [clogb2(RAM_DEPTH-1)-1:0] addra,  // Port A address bus, width determined from RAM_DEPTH
  input [clogb2(RAM_DEPTH-1)-1:0] addrb,  // Port B address bus, width determined from RAM_DEPTH
  input [RAM_WIDTH-1:0] dina,           // Port A RAM input data
  input clk,                           // Clock
  input wea,                            // Port A write enable
  input ena,                            // Port A RAM Enable, for additional power savings, disable port when not in use
  input enb,                            // Port B RAM Enable, for additional power savings, disable port when not in use
  input rst,                           // Port A and B output reset (does not affect memory contents)
  input regceb,                         // Port B output register enable
  
  output [RAM_WIDTH-1:0] doutb                   // Port B RAM output data
    );

`ifdef BRAM                     // Condition check for macro 
   
	BRAM_singlePort_readFirst #(
	  RAM_WIDTH,                  
	  RAM_DEPTH,                  
	  RAM_PERFORMANCE, 
	  INIT_FILE                      
	)
		bram
	(
	  addra,  
	  addrb, 
	  dina,   
	  clk,
	  wea,
	  ena,
	  enb,
	  rst, 
	  regceb, 
	  doutb 
		);
	initial $display("defined");

`else
 		
  wire [13:0] addr;
  assign addr = wea?{{(14-clogb2(RAM_DEPTH-1)){1'b0}},addra}:{{(14-clogb2(RAM_DEPTH-1)){1'b0}},addrb};	
	
  wire [RAM_WIDTH-1:0] doutb_int; 

  SB_SPRAM256KA spram
  (
    .ADDRESS(addr),
    .DATAIN(dina),
    .MASKWREN({wea, wea, wea, wea}),
    .WREN(wea),
    .CHIPSELECT(1'b1),
    .CLOCK(clk),
    .STANDBY(1'b0),
    .SLEEP(1'b0),
    .POWEROFF(1'b1),
    .DATAOUT(doutb_int)
  );
	
  reg [RAM_WIDTH-1:0] doutb_reg;	
  always @(posedge clk)
	doutb_reg <= doutb_int;

  assign doutb = doutb_reg;	

  initial $display("not defined");

`endif 

  //  The following function calculates the address width based on specified RAM depth
  function integer clogb2;
    input integer depth;
      for (clogb2=0; depth>0; clogb2=clogb2+1)
        depth = depth >> 1;
  endfunction

endmodule

`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 28.10.2022 17:15:24
// Design Name: 
// Module Name: accumulator
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

// This code implements a parameterizable subtractor followed by multiplier which will be packed into DSP Block
(* use_dsp = "yes" *)
module accumulator
#(
parameter SIZEIN = 16  // Size of inputs
)
(
input clk, // Clock
input ce1,ce2,ce3,  // Clock enable
input rst, // Reset
input clear_and_go, clear, // to reset the accumulator
input signed [SIZEIN-1:0] a,  // 1st Input to pre-subtractor
input signed [SIZEIN-1:0] b,  // 2nd input to pre-subtractor
//input signed [SIZEIN-1:0] c,  // multiplier input
output signed [2*SIZEIN-1:0] presubmult_out
);
    


// Declare registers for intermediate values
reg signed [SIZEIN-1:0] a_reg, b_reg; 
//reg signed [SIZEIN-1:0]c_reg;
reg signed [SIZEIN:0]   add_reg;
//reg signed [2*SIZEIN:0] m_reg;
reg signed [2*SIZEIN:0] p_reg;

always @(posedge clk)
 if (rst | clear)
  begin
    a_reg   <= 0;
    b_reg   <= 0;
    //c_reg   <= 0;
	add_reg <= 0;
    //m_reg   <= 0;
    p_reg   <= 0;
  end
 else begin
    if (ce1)
      begin
        a_reg   <= a;
        b_reg   <= b;
        //c_reg   <= c;
      end
    if (ce2)
        add_reg <= a_reg + b_reg;
    //m_reg   <= add_reg * c_reg;
    //p_reg   <= p_reg + m_reg;
    if (ce3)
        if(clear_and_go)
            p_reg <= add_reg;
        else    
            p_reg   <= p_reg + add_reg;
  end

assign presubmult_out = p_reg;    
    

endmodule

`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 28.10.2022 16:14:22
// Design Name: 
// Module Name: adder_simd
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

// This module describes SIMD Inference 
// 4 small adders can be packed into signle DSP block

// Note : SV constructs are used, Compile this with System Verilog Mode


// Apply this attribute on the module definition
(* use_dsp = "simd" *)
module adder_simd
    #(
    parameter N = 2,    // Number of Adders
    parameter W = 15   // Width of the Adders
    )
    (
    input clk, en,
    input [W-1:0] a_0, a_1,
    input [W-1:0] b_0, b_1,
    output reg signed [W:0] out_0,
    output reg signed [W:0] out_1
    );


                   
integer i;
logic signed [W-1:0] a_r [N-1:0];
logic signed [W-1:0] b_r [N-1:0];

always @ (posedge clk)
     if(en)
      begin 
      a_r[0] <= a_0;
      b_r[0] <= b_0;
      out_0 <= a_r[0] + b_r[0];
      a_r[1] <= a_1;
      b_r[1] <= b_1;
      out_1 <= a_r[1] + b_r[1];
      end   

endmodule

`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03.11.2022 10:58:43
// Design Name: 
// Module Name: fifo
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module bram_fifo#(
parameter DATA_WIDTH = 25, DEPTH = 256
)
(
input clk, rst,
input [DATA_WIDTH-1:0] DI,
input rden, wren,
output [DATA_WIDTH-1:0] DO
`ifdef CONFIGURABILITY
	, input clear_counter
`endif
    );

reg [clogb2(DEPTH-1)-1:0] rd_cnt;
reg [clogb2(DEPTH-1)-1:0] wr_cnt;

always @(posedge clk)
    if(rst)
        rd_cnt <= 0;
	`ifdef CONFIGURABILITY
	else if(clear_counter)
		rd_cnt <= 0;
	`endif
		else if(rden)
		    if(rd_cnt < DEPTH-1)
		        rd_cnt <= rd_cnt + 1'b1;
		    else
		        rd_cnt <= 0;

always @(posedge clk)
    if(rst)
        wr_cnt <= 0;
	`ifdef CONFIGURABILITY
	else if(clear_counter)
			wr_cnt <= 0;	
	`endif
		else if(wren)
		    if(wr_cnt < DEPTH-1)
		        wr_cnt <= wr_cnt + 1'b1;
		    else
		        wr_cnt <= 0;

BRAM_singlePort_readFirst
#(
  .RAM_WIDTH(DATA_WIDTH),          // Specify RAM data width
  .RAM_DEPTH(DEPTH),               // Specify RAM depth (number of entries)
  .RAM_PERFORMANCE("LOW_LATENCY"), // Select "HIGH_PERFORMANCE" or "LOW_LATENCY" 
  .INIT_FILE("")                   // Specify name/location of RAM initialization file if using one (leave blank if not)
  )
fifo_ram
 (
  .addra(wr_cnt),                  // Port A address bus, driven by axi bus
  .addrb(rd_cnt),                  // Port B address bus, it goes in the accumulator
  .dina(DI),                       // Port A RAM input data, driven by axi bus
  .clk(clk),                       // Clock
  .wea(wren),                      // Port A write enable
  .ena(wren),                      // Port A RAM Enable, for additional power savings, disable port when not in use
  .enb(rden),                      // Port B RAM Enable, for additional power savings, disable port when not in use
  .rst(rst),                       // Port A and B output reset (does not affect memory contents)
  .regceb(1'b1),                   // Port B output register enable
  
  .doutb(DO)              // Port B RAM output data
    );


////////////////////////////
//  _               ____  //
// | | ___   __ _  |___ \ //
// | |/ _ \ / _` |   __)  //
// | | (_) | (_| |  / __/ //
// |_|\___/ \__, | |_____ //
//          |___/         //
////////////////////////////
   
//  The following function calculates the address width based on specified RAM depth
function integer clogb2;
  input integer depth;
    for (clogb2=0; depth>0; clogb2=clogb2+1)
      depth = depth >> 1;
endfunction   

endmodule


`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03.11.2022 12:02:52
// Design Name: 
// Module Name: conv
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////



module conv
    #(
    parameter WEIGHTS = 4,     // Number of addends
    parameter DATA_WIDTH = 15 // Width of the Adders
    )
(
    input clk, rst, en, acc_clear_and_go, acc_clear,
    input signed [WEIGHTS*DATA_WIDTH-1:0] weights_in,
    input [WEIGHTS-1:0] spikes,
    
    output [2*(DATA_WIDTH+1)-1:0] out,
    output valid
    );
    
localparam SUM = WEIGHTS/2;
localparam PIPE = 5; // pipeline stages

wire signed [DATA_WIDTH-1:0] weights [WEIGHTS-1:0];
assign weights[0] = weights_in[DATA_WIDTH-1:0];
assign weights[1] = weights_in[2*DATA_WIDTH-1:DATA_WIDTH];
assign weights[2] = weights_in[3*DATA_WIDTH-1:2*DATA_WIDTH];
assign weights[3] = weights_in[4*DATA_WIDTH-1:3*DATA_WIDTH];

// enable shift
reg [PIPE-1:0] en_shift;
integer i;
always @(posedge clk)
    if (rst)
        en_shift <= 0;
    else
        begin
        en_shift[0] <= en;
        for(i=1;i<PIPE;i=i+1)
            en_shift[i] <= en_shift[i-1];
        end

// weights sorting
wire signed [DATA_WIDTH-1:0] weights_a [SUM-1:0]; 
wire signed [DATA_WIDTH-1:0] weights_b [SUM-1:0];  
assign weights_a[0] = weights[2];
assign weights_a[1] = weights[3];
assign weights_b[0] = weights[0];
assign weights_b[1] = weights[1];
// spike sorting (no that one)
wire [DATA_WIDTH-1:0] spike_a [SUM-1:0]; 
wire [DATA_WIDTH-1:0] spike_b [SUM-1:0]; 
assign spike_a[1] = {(DATA_WIDTH){spikes[WEIGHTS-1]}};     
assign spike_a[0] = {(DATA_WIDTH){spikes[WEIGHTS-2]}};     
assign spike_b[1] = {(DATA_WIDTH){spikes[WEIGHTS-3]}};     
assign spike_b[0] = {(DATA_WIDTH){spikes[WEIGHTS-4]}};     

// conv(weights*spike)
wire [DATA_WIDTH-1:0] anded_weights_a [SUM-1:0];     
wire [DATA_WIDTH-1:0] anded_weights_b [SUM-1:0];     
assign anded_weights_a[0] = weights_a[0] & spike_a[0];
assign anded_weights_a[1] = weights_a[1] & spike_a[1];
assign anded_weights_b[0] = weights_b[0] & spike_b[0];
assign anded_weights_b[1] = weights_b[1] & spike_b[1];

// simd adder
wire signed [DATA_WIDTH:0] double_adder_out [SUM-1:0];
wire double_adder_en;
assign double_adder_en = en | en_shift[0];
adder_simd #(SUM, DATA_WIDTH)
    double_adder
    (
    clk,
    double_adder_en,
    anded_weights_a[0],
    anded_weights_a[1],
    anded_weights_b[0],
    anded_weights_b[1],
    double_adder_out[0],
    double_adder_out[1]
    );

// accumulator
wire [2*(DATA_WIDTH+1)-1:0] acc_out;
accumulator
#(DATA_WIDTH+1)
acc
(
clk,en_shift[1],en_shift[2],en_shift[3],rst,acc_clear_and_go,acc_clear,double_adder_out[0],double_adder_out[1],acc_out
);

            
// output assignment
assign valid = en_shift[PIPE-1];
assign out = valid? acc_out:0;    

endmodule

`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03.11.2022 14:26:36
// Design Name: 
// Module Name: integrator
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module integrator #(parameter WIDTH = 25)
    (
    input clk, rst, en, detection,
    input [WIDTH-1:0] output_old,
    input [13:0] decay,
    input [WIDTH-1:0] stimolo,
    input [WIDTH-1:0] threshold,
    
    output valid,
    output spike,
    output [WIDTH-1:0] output_new
    );

localparam P_SIZE = WIDTH+13;
    
reg signed [WIDTH-1:0] r_output_old;  
reg signed [13:0] r_decay; 
reg signed [P_SIZE-1:0] p;
reg signed [WIDTH-1:0] p_shift;
reg signed [WIDTH-1:0] r_stimolo[2:0];
reg signed [WIDTH-1:0] r_threshold [3:0];
(* use_dsp = "yes" *)
reg signed [WIDTH-1:0] comparator_in;
reg [3:0] en_shift;

reg [P_SIZE-1:0] supporto_1, supporto_2;

integer i;
always @(posedge clk)
    if(rst) begin
        en_shift <= 0;
        r_output_old <= 0;
        r_decay <= 0;
        p <= 0;
        p_shift <= 0;
        comparator_in <= 0;
         for(i=0;i<4;i=i+1)
            r_threshold[i] <= 0;
        
        for(i=0;i<3;i=i+1)
            r_stimolo[i] <= 0;
    end
    else begin
        en_shift[0] <= en;
        r_output_old <= output_old;
        r_decay <= decay;
        r_stimolo[0] <= stimolo;
        r_threshold[0] <= threshold;
        
        en_shift[1] <= en_shift[0];
        p <= r_output_old*r_decay;
        r_stimolo[1] <= r_stimolo[0];
        r_threshold[1] <= r_threshold[0];
        
        en_shift[2] <= en_shift[1];
        if(p[P_SIZE-1]) begin
            supporto_1 = ~p+1'b1;            
            supporto_2 = supporto_1[P_SIZE-1:12];
            p_shift <= ~supporto_2+1'b1;
        end
        else
            p_shift <= p[P_SIZE-1:12];
        
        r_stimolo[2] <= r_stimolo[1];
        r_threshold[2] <= r_threshold[1];
        
        en_shift[3] <= en_shift[2];
        comparator_in <= p_shift + r_stimolo[2];
        r_threshold[3] <= r_threshold[2];
    end

assign spike = (comparator_in >= r_threshold[3]) & detection? 1'b1 : 1'b0; 
assign output_new = spike? 0 : comparator_in;     
assign valid = en_shift[3];
    
endmodule

`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03.11.2022 14:26:36
// Design Name: 
// Module Name: integrator
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module integrator_and_fifo #(parameter DEPTH = 256, parameter WIDTH = 25)
    (
    input clk, rst, en, detection,
    input [13:0] decay,
    input [WIDTH-1:0] stimolo,
    input [WIDTH-1:0] threshold,
    
    output valid,
    output spike,
    output [WIDTH-1:0] output_new
	`ifdef CONFIGURABILITY
		,input clear_counter
	`endif
    );
    
    wire [WIDTH-1:0] output_old;
    
    // FIFO
    bram_fifo #( .DATA_WIDTH(WIDTH), .DEPTH(DEPTH) ) fifo_i (.clk(clk),.rst(rst),.DI(output_new),.rden(en),.wren(valid),.DO(output_old) `ifdef CONFIGURABILITY , .clear_counter(clear_counter) `endif); 
    // wait fifo output
    reg en_d;
    reg [WIDTH-1:0] stimolo_d;
    always @(posedge clk)
        if (rst) begin
            en_d <= 0;
            stimolo_d <= 0;
         end
        else begin 
            en_d <= en;
            if(en)
                stimolo_d <= stimolo;
        end
    // INTEGRATOR
    integrator #(.WIDTH(WIDTH)) integrator_i (clk, rst, en_d, detection,output_old,decay,stimolo_d,threshold,valid,spike,output_new);
    

    endmodule

`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03.04.2023 11:05:04
// Design Name: 
// Module Name: layer_lp
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module layer_lp
    #(
    parameter WIDTH = 25,
    parameter NEURON = 256,   
	parameter LAYERS = 4,
    parameter WEIGHTS_FILE_1 = "weights_1.txt",
	parameter WEIGHTS_FILE_2 = "weights_2.txt",
    parameter [13:0] current_decay_1 = 4096 - 4096,
	parameter [13:0] current_decay_2 = 4096 - 4096,
	parameter [13:0] current_decay_3 = 4096 - 4096,
	parameter [13:0] current_decay_4 = 4096 - 4096,
    parameter [13:0] voltage_decay_1 = 4096 - 415,
	parameter [13:0] voltage_decay_2 = 4096 - 415,
	parameter [13:0] voltage_decay_3 = 4096 - 415,
	parameter [13:0] voltage_decay_4 = 4096 - 415,
    parameter [WIDTH-1:0]  threshold_1 = 6,
    parameter [WIDTH-1:0]  threshold_2 = 6,
    parameter [WIDTH-1:0]  threshold_3 = 6,
    parameter [WIDTH-1:0]  threshold_4 = 6,
    parameter WEIGHT_DEPTH = 8192
    )
    (
    input clk, rst,
    input en,
    input [3:0] spike_in,
    input active_group_in,

	input [clogb2(WEIGHT_DEPTH-1)-1:0] weight_rd_addr,
	input acc_clear, acc_clear_and_go,
	output convolution_pipe_full,    
	input [clogb2(LAYERS-1)-1:0] layer_id,

    output valid,
    output [1:0] spike_out,
    output active_group_out,
    output reg valid_potential,
    output signed [WIDTH-1:0] neuron_lp_voltage,
	output integrated_neuron,
    
    input [7:0] weight_mem_L1_wren,
    input [clogb2(WEIGHT_DEPTH-1)-1:0] weight_mem_L1_wr_addr,
    input [16-1:0] weight_mem_L1_data_in,
    output [16-1:0] weight_mem_L1_data_out,
    input weight_mem_L1_ena,
    input [7:0] weight_mem_L2_wren,
    input [clogb2(WEIGHT_DEPTH-1)-1:0] weight_mem_L2_wr_addr,
    input [16-1:0] weight_mem_L2_data_in,
    output [16-1:0] weight_mem_L2_data_out,
    input weight_mem_L2_ena,
	output [7:0] weight_debug,
	output weight_en_debug
    );

assign weight_en_debug = en;
assign weight_debug = weight_rd_addr[7:0];//weights_out_1[7:0];

/////////////////////////////////////////////////////////////////
//               _       _     _                               //
// __      _____(_) __ _| |__ | |_   _ __ ___   ___ _ __ ___   //
// \ \ /\ / / _ \ |/ _` | '_ \| __| | '_ ` _ \ / _ \ '_ ` _ \  //
//  \ V  V /  __/ | (_| | | | | |_  | | | | | |  __/ | | | | | //
//   \_/\_/ \___|_|\__, |_| |_|\__| |_| |_| |_|\___|_| |_| |_| //
//                 |___/                                       //
/////////////////////////////////////////////////////////////////   

localparam WEIGHT = 8;  

wire [15:0] weights_out_1;   
wire [15:0] weights_out_2;   

SPRAM_singlePort_readFirst
#(
  .RAM_WIDTH(16),                  // Specify RAM data width
  .RAM_DEPTH(WEIGHT_DEPTH),             // Specify RAM depth (number of entries)
  .RAM_PERFORMANCE("HIGH_PERFORMANCE"), // Select "HIGH_PERFORMANCE" or "LOW_LATENCY" 
  .INIT_FILE(WEIGHTS_FILE_1)             // Specify name/location of RAM initialization file if using one (leave blank if not)
  )
weight_mem_1
 (
  .addra(weight_mem_L1_wr_addr),      // Port A address bus, driven by axi bus
  .addrb(weight_rd_addr),          // Port B address bus, it goes in the accumulator
  .dina(weight_mem_L1_data_in),       // Port A RAM input data, driven by axi bus
  //.dinb({64{1'b0}}),                        // write port PL side not used
  .clk(clk),                       // Clock
  .wea(weight_mem_L1_wren[0]),           // Port A write enable
  //.web(1'b0),                      // write port PL side not used
  .ena(weight_mem_L1_ena),         // Port A RAM Enable, for additional power savings, disable port when not in use
  .enb(1'b1),                      // Port B RAM Enable, for additional power savings, disable port when not in use
  .rst(rst),                       // Port A and B output reset (does not affect memory contents)
  //.regcea(1'b1),                   // Port A output register enable
  .regceb(1'b1),                   // Port B output register enable
  
  //.douta(weights_mem_ctrl_ext_1),    // Port B RAM output data
  .doutb(weights_out_1)              // Port B RAM output data
    );

SPRAM_singlePort_readFirst
#(
  .RAM_WIDTH(16),                  // Specify RAM data width
  .RAM_DEPTH(WEIGHT_DEPTH),             // Specify RAM depth (number of entries)
  .RAM_PERFORMANCE("HIGH_PERFORMANCE"), // Select "HIGH_PERFORMANCE" or "LOW_LATENCY" 
  .INIT_FILE(WEIGHTS_FILE_2)             // Specify name/location of RAM initialization file if using one (leave blank if not)
  )
weight_mem_2
 (
  .addra(weight_mem_L2_wr_addr),      // Port A address bus, driven by axi bus
  .addrb(weight_rd_addr),          // Port B address bus, it goes in the accumulator
  .dina(weight_mem_L2_data_in),       // Port A RAM input data, driven by axi bus
  //.dinb({64{1'b0}}),                        // write port PL side not used
  .clk(clk),                       // Clock
  .wea(weight_mem_L2_wren[0]),           // Port A write enable
  //.web(1'b0),                      // write port PL side not used
  .ena(weight_mem_L2_ena),                      // Port A RAM Enable, for additional power savings, disable port when not in use
  .enb(1'b1),                      // Port B RAM Enable, for additional power savings, disable port when not in use
  .rst(rst),                       // Port A and B output reset (does not affect memory contents)
  //.regcea(1'b1),                   // Port A output register enable
  .regceb(1'b1),                   // Port B output register enable
  
  //.douta(weights_mem_ctrl_ext_2),    // Port B RAM output data
  .doutb(weights_out_2)              // Port B RAM output data
    );

wire signed [31:0] weights;
assign weights = {weights_out_1,weights_out_2};

///////////////////////////////////////////////////////////
//                            _       _   _              //
//   ___ ___  _ ____   _____ | |_   _| |_(_) ___  _ __   //
//  / __/ _ \| '_ \ \ / / _ \| | | | | __| |/ _ \| '_ \  //
// | (_| (_) | | | \ V / (_) | | |_| | |_| | (_) | | | | //
//  \___\___/|_| |_|\_/ \___/|_|\__,_|\__|_|\___/|_| |_| //
//  _____                               _ _              //
//    __/  __      __ __  __  ___ _ __ (_) | _____       //
//  \ \    \ \ /\ / / \ \/ / / __| '_ \| | |/ / _ \      //
//  / /__   \ V  V /   >  <  \__ \ |_) | |   <  __/      //
//  _____\   \_/\_/   /_/\_\ |___/ .__/|_|_|\_\___|      //
//                               |_|                     //
///////////////////////////////////////////////////////////

// Spikes and weights are convolved: multiplied (by logic-and) and accumulated (5-way accumulator)

wire [2*(WEIGHT+1)-1:0] stimulus;

conv
    #(
    .WEIGHTS(4),          // Number of addends
    .DATA_WIDTH(WEIGHT)   // Width of the Adders
    )
 conv_i
(
    .clk(clk), .rst(rst), .en(en), .acc_clear_and_go(acc_clear_and_go), .acc_clear(acc_clear),
    .weights_in(weights),
    .spikes(spike_in),
    
    .out(stimulus),
    .valid(convolution_pipe_full)
    );

//////////////////////////////////////////////
//     _   _                                //
//    | \ | | ___ _   _ _ __ ___  _ __      //
//    |  \| |/ _ \ | | | '__/ _ \| '_ \     //
//    | |\  |  __/ |_| | | | (_) | | | |    //
//    |_| \_|\___|\__,_|_|  \___/|_| |_|    //
//    ___                  _   ____  ____   //
//   |  _|  __ _ _ __   __| | / ___||  _ \  //
//   | |   / _` | '_ \ / _` | \___ \| | | | //
//  _| |  | (_| | | | | (_| |  ___) | |_| | //
// |___|   \__,_|_| |_|\__,_| |____/|____/  //
//                                          //
//////////////////////////////////////////////

neuron_lp 
    #(.DEPTH(NEURON),.WIDTH(WIDTH),.WEIGHT(WEIGHT),.LAYERS(LAYERS))
 neuron_lp_i
    (
    .clk(clk), .rst(rst), .en(acc_clear_and_go),
    .current_decay_1(current_decay_1), .voltage_decay_1(voltage_decay_1), .threshold_1(threshold_1),
    .current_decay_2(current_decay_2), .voltage_decay_2(voltage_decay_2), .threshold_2(threshold_2),
	.current_decay_3(current_decay_3), .voltage_decay_3(voltage_decay_3), .threshold_3(threshold_3),
    .current_decay_4(current_decay_4), .voltage_decay_4(voltage_decay_4), .threshold_4(threshold_4),    
	.synaptic_current(stimulus),
    .layer_id(layer_id),
    
    .valid(valid),
    .spike_p(spike_out),
    .active_group(active_group_out),
    .voltage_ready(integrated_neuron),
    .voltage(neuron_lp_voltage)
    );


////////////////////////////
//  _               ____  //
// | | ___   __ _  |___ \ //
// | |/ _ \ / _` |   __)  //
// | | (_) | (_| |  / __/ //
// |_|\___/ \__, | |_____ //
//          |___/         //
////////////////////////////
   
//  The following function calculates the address width based on specified RAM depth
function integer clogb2;
  input integer depth;
    for (clogb2=0; depth>0; clogb2=clogb2+1)
      depth = depth >> 1;
endfunction      

function integer max;
  input integer a,b;
    if (a>b)
      max = a;
    else
      max = b;
endfunction 
    
endmodule


`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03.04.2023 12:53:59
// Design Name: 
// Module Name: neuron_lp
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module neuron_lp#(parameter DEPTH = 256, parameter WIDTH = 25, parameter WEIGHT = 8, parameter LAYERS = 4)
    (
    input clk, rst, en,
    input [13:0] current_decay_1, voltage_decay_1,
    input [WIDTH-1:0] threshold_1,
    input [13:0] current_decay_2, voltage_decay_2,
    input [WIDTH-1:0] threshold_2,
	input [13:0] current_decay_3, voltage_decay_3,
    input [WIDTH-1:0] threshold_3,
    input [13:0] current_decay_4, voltage_decay_4,
    input [WIDTH-1:0] threshold_4,
    input [2*(WEIGHT+1)-1:0] synaptic_current,
    input [clogb2(LAYERS-1)-1:0] layer_id,
    
    output valid,
    output [1:0] spike_p,
    output active_group,
    output voltage_ready,
    output [WIDTH-1:0] voltage
    );

localparam CURRENT_WIDTH = 2*(WEIGHT+1);

wire [13:0] current_decay, voltage_decay;
wire [WIDTH-1:0] threshold;

assign current_decay = layer_id == 0 ? current_decay_1: 
					   layer_id == 1 ? current_decay_2:
					   layer_id == 2 ? current_decay_3: 
					   				   current_decay_4;

assign voltage_decay = layer_id == 0 ? voltage_decay_1: 
					   layer_id == 1 ? voltage_decay_2:
					   layer_id == 2 ? voltage_decay_3: 
					   				   voltage_decay_4;

assign threshold     = layer_id == 0 ? threshold_1: 
					   layer_id == 1 ? threshold_2:
					   layer_id == 2 ? threshold_3: 
					   			       threshold_4;

`ifdef CONFIGURABILITY
	reg [clogb2(LAYERS-1)-1:0] layer_id_d;
	always @(posedge clk)
			layer_id_d <= layer_id;

	wire clear_counter;
	assign clear_counter = layer_id_d > layer_id;
`endif

wire [WIDTH-1:0] current;
wire [WIDTH-1:0] synaptic_current_ext;
wire spike_s;
//assign synaptic_current_ext = {{(WIDTH-CURRENT_WIDTH-7){synaptic_current[CURRENT_WIDTH-1]}},synaptic_current,{(7){1'b0}}};
    
//integrator_and_fifo #(.DEPTH(DEPTH), .WIDTH(WIDTH)) Current_i (clk, rst, en, 1'b0, current_decay,synaptic_current_ext,threshold,current_ready,current_spike, current);   
//integrator_and_fifo #(.DEPTH(DEPTH), .WIDTH(WIDTH)) Voltage_i (clk, rst, current_ready, 1'b1, voltage_decay,current,threshold,voltage_ready,spike_s,voltage);   
integrator_and_fifo #(.DEPTH(DEPTH), .WIDTH(WIDTH)) Voltage_i (clk, rst, en, 1'b1, voltage_decay,synaptic_current,threshold,voltage_ready,spike_s,voltage `ifdef CONFIGURABILITY , clear_counter `endif);   
s2p #(2) s2p_i (clk, rst, voltage_ready, spike_s, spike_p, valid, active_group);    
   
////////////////////////////
//  _               ____  //
// | | ___   __ _  |___ \ //
// | |/ _ \ / _` |   __)  //
// | | (_) | (_| |  / __/ //
// |_|\___/ \__, | |_____ //
//          |___/         //
////////////////////////////
   
//  The following function calculates the address width based on specified RAM depth
function integer clogb2;
  input integer depth;
    for (clogb2=0; depth>0; clogb2=clogb2+1)
      depth = depth >> 1;
endfunction     

endmodule

`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03.04.2023 11:05:04
// Design Name: 
// Module Name: snn_lp
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module snn_lp 
#(
    // neuron arithmetic
    parameter WIDTH = 26,
	// 
	parameter MAX_SYNAPSES  = 256,
	parameter MAX_NEURONS = 256,
	parameter LAYERS = 4,
    // weight mem 1
    parameter INPUT_SPIKE_1 = 128, 
    parameter NEURON_1 = 64,   
    parameter WEIGHTS_FILE_1 = "weights_1.txt",
    parameter [13:0] current_decay_1 = 4096 - 4096,
    parameter [13:0] voltage_decay_1 = 4096 - 415,
    parameter [WIDTH-1:0]  threshold_1 = 6,
    // weight mem 2
    parameter INPUT_SPIKE_2 = 64,
    parameter NEURON_2 = 128,
    parameter WEIGHTS_FILE_2 = "weights_2.txt",
    parameter [13:0] current_decay_2 = 4096 - 4096,
    parameter [13:0] voltage_decay_2 = 4096 - 424,
    parameter [WIDTH-1:0]  threshold_2 = 6,
    // weight mem 3
    parameter INPUT_SPIKE_3 = 128, 
    parameter NEURON_3 = 256,   
    parameter WEIGHTS_FILE_3 = "weights_3.txt",
    parameter [13:0] current_decay_3 = 4096 - 4096,
    parameter [13:0] voltage_decay_3 = 4096 - 415,
    parameter [WIDTH-1:0]  threshold_3 = 6,
    // weight mem 4
    parameter INPUT_SPIKE_4 = 256,
    parameter NEURON_4 = 128,
    parameter WEIGHTS_FILE_4 = "weights_4.txt",
    parameter [13:0] current_decay_4 = 4096 - 4096,
    parameter [13:0] voltage_decay_4 = 4096 - 424,
    parameter [WIDTH-1:0]  threshold_4 = 6,
    
    parameter WEIGHT_DEPTH_12 = 8192,
    parameter WEIGHT_DEPTH_34 = 8192
)
    (
    // input
    input clk, rst,
    input en,
    input [3:0] spike_in,
    input active_group_in,

    // output
    output valid,	
	output valid_spike,
    output [3:0] spike_out,
	output integrated_neuron,

    // weight mem 1
    input [7:0] weight_mem_L1_wren,
    input [clogb2(WEIGHT_DEPTH_12-1)-1:0] weight_mem_L1_wr_addr,
    input [16-1:0] weight_mem_L1_data_in,
    output [16-1:0] weight_mem_L1_data_out,
    input weight_mem_L1_ena,

    // weight mem 2
    input [7:0] weight_mem_L2_wren,
    input [clogb2(WEIGHT_DEPTH_12-1)-1:0] weight_mem_L2_wr_addr,
    input [16-1:0] weight_mem_L2_data_in,
    output [16-1:0] weight_mem_L2_data_out,
    input weight_mem_L2_ena,

    // weight mem 3
    input [7:0] weight_mem_L3_wren,
    input [clogb2(WEIGHT_DEPTH_34-1)-1:0] weight_mem_L3_wr_addr,
    input [16-1:0] weight_mem_L3_data_in,
    output [16-1:0] weight_mem_L3_data_out,
    input weight_mem_L3_ena,

    // weight mem 4
    input [7:0] weight_mem_L4_wren,
    input [clogb2(WEIGHT_DEPTH_34-1)-1:0] weight_mem_L4_wr_addr,
    input [16-1:0] weight_mem_L4_data_in,
    output [16-1:0] weight_mem_L4_data_out,
    input weight_mem_L4_ena,

	//spike mem 1 & 2
	output wire [7:0] o_spike_mem_dat,
	input wire [7:0] i_spike_mem_adr,
	input wire [1:0] i_spike_mem_rd_en,
	input wire [1:0] i_spike_mem_wr_en,
	input wire [3:0] i_spike_mem_dat,

	// # layers and channels
	input wire [clogb2(MAX_SYNAPSES-1)-1:0] snn_input_channels, 
	input wire [clogb2(MAX_NEURONS-1)-1:0] neuron_1, neuron_2, neuron_3, neuron_4, 
	input wire [2:0] layers,

	// output buffer access
	input output_buffer_ren,
	input [7:0] output_buffer_addr,
	output [31:0] output_buffer_out
    );

localparam LAYERS_LOG2 = clogb2(LAYERS-1);
`ifdef CONFIGURABILITY
	localparam TOTAL_NEURONS = 4*MAX_NEURONS; 
`else
	localparam TOTAL_NEURONS = NEURON_1 + NEURON_2 + NEURON_3 + NEURON_4;
`endif

localparam SYN_G1_L2 = clogb2(INPUT_SPIKE_1/4-1);
localparam SYN_G2_L2 = clogb2(INPUT_SPIKE_2/4-1);
localparam SYN_G3_L2 = clogb2(INPUT_SPIKE_3/4-1);
localparam SYN_G4_L2 = clogb2(INPUT_SPIKE_4/4-1);

localparam MAX_LAYER = max4(INPUT_SPIKE_1*NEURON_1,INPUT_SPIKE_2*NEURON_2,INPUT_SPIKE_3*NEURON_3,INPUT_SPIKE_4*NEURON_4)/8;

localparam P1 = clogb2(MAX_LAYER)-clogb2(NEURON_1*INPUT_SPIKE_1/8);
localparam P2 = clogb2(MAX_LAYER)-clogb2(NEURON_2*INPUT_SPIKE_2/8);
localparam P3 = clogb2(MAX_LAYER)-clogb2(NEURON_3*INPUT_SPIKE_3/8);
localparam P4 = clogb2(MAX_LAYER)-clogb2(NEURON_4*INPUT_SPIKE_4/8);

localparam WEIGHT_ADDRESS_SIZE = clogb2(MAX_NEURONS/2-1) + clogb2(MAX_SYNAPSES/4-1) + clogb2(LAYERS-1);

// interconnection signals
wire v1,v2;
wire [1:0] s1,s2;
wire ag1,ag2;

/*
  _     ____            _     _ 
 | |   |  _ \          | |   / |
 | |   | |_) |  _____  | |   | |
 | |___|  __/  |_____| | |___| |
 |_____|_|             |_____|_|
                                
*/    
wire signed [WIDTH-1:0] voltage_1;
layer_lp
    #(
    .WIDTH(WIDTH),
    .NEURON(TOTAL_NEURONS/2),   
	.LAYERS(LAYERS),
    .WEIGHTS_FILE_1(WEIGHTS_FILE_1),
	.WEIGHTS_FILE_2(WEIGHTS_FILE_2),
    .current_decay_1(current_decay_1),
	.current_decay_2(current_decay_2),
	.current_decay_3(current_decay_3),
	.current_decay_4(current_decay_4),
    .voltage_decay_1(voltage_decay_1),
	.voltage_decay_2(voltage_decay_2),
	.voltage_decay_3(voltage_decay_3),
	.voltage_decay_4(voltage_decay_4),
    .threshold_1(threshold_1),
	.threshold_2(threshold_2),
	.threshold_3(threshold_3),
	.threshold_4(threshold_4),    
    .WEIGHT_DEPTH(2**(WEIGHT_ADDRESS_SIZE))
    )
layer_lp_l1_i
    (
    .clk(clk), .rst(rst),
    .en(layer_enable_dd),
    .spike_in(spike_mem_out),
    .active_group_in(),
    
	.weight_rd_addr(weight_rd_addr),
	.acc_clear(layer_integrated), .acc_clear_and_go(convolution_valid),
	.convolution_pipe_full(convolution_pipe_full),    
	.layer_id(layer_counter),

    .valid(v1),
    .spike_out(s1),
    .active_group_out(ag1),
    .integrated_neuron(integrated_neuron_1),
    .neuron_lp_voltage(voltage_1),

   .weight_mem_L1_wren(weight_mem_L1_wren),
   .weight_mem_L1_wr_addr(weight_mem_L1_wr_addr),
   .weight_mem_L1_data_in(weight_mem_L1_data_in),
   .weight_mem_L1_data_out(weight_mem_L1_data_out),
   .weight_mem_L1_ena(weight_mem_L1_ena),
   .weight_mem_L2_wren(weight_mem_L2_wren),
   .weight_mem_L2_wr_addr(weight_mem_L2_wr_addr),
   .weight_mem_L2_data_in(weight_mem_L2_data_in),
   .weight_mem_L2_data_out(weight_mem_L2_data_out),
   .weight_mem_L2_ena(weight_mem_L2_ena)

	//.weight_debug(weight_debug),
	//.weight_en_debug(weight_en_debug)
    );    
/*
  _     ____            _     ____  
 | |   |  _ \          | |   |___ \ 
 | |   | |_) |  _____  | |     __) |
 | |___|  __/  |_____| | |___ / __/ 
 |_____|_|             |_____|_____|
                                    
*/
wire signed [WIDTH-1:0] voltage_2;
layer_lp
    #(
    .WIDTH(WIDTH),
    .NEURON(TOTAL_NEURONS/2),      
	.LAYERS(LAYERS),
    .WEIGHTS_FILE_1(WEIGHTS_FILE_3),
	.WEIGHTS_FILE_2(WEIGHTS_FILE_4),
    .current_decay_1(current_decay_1),
	.current_decay_2(current_decay_2),
	.current_decay_3(current_decay_3),
	.current_decay_4(current_decay_4),
    .voltage_decay_1(voltage_decay_1),
	.voltage_decay_2(voltage_decay_2),
	.voltage_decay_3(voltage_decay_3),
	.voltage_decay_4(voltage_decay_4),
    .threshold_1(threshold_1),
	.threshold_2(threshold_2),
	.threshold_3(threshold_3),
	.threshold_4(threshold_4),    
    .WEIGHT_DEPTH(2**(WEIGHT_ADDRESS_SIZE))
    )
layer_lp_l2_i
    (
    .clk(clk), .rst(rst),
    .en(layer_enable_dd),
    .spike_in(spike_mem_out),
    .active_group_in(),
    
	.weight_rd_addr(weight_rd_addr),
	.acc_clear(layer_integrated), .acc_clear_and_go(convolution_valid),
	.convolution_pipe_full(convolution_pipe_full),    
	.layer_id(layer_counter),    
    
    .valid(v2),
    .spike_out(s2),
    .active_group_out(ag2),
    .neuron_lp_voltage(voltage_2),
	.integrated_neuron(integrated_neuron_2),
    
   .weight_mem_L1_wren(weight_mem_L3_wren),
   .weight_mem_L1_wr_addr(weight_mem_L3_wr_addr),
   .weight_mem_L1_data_in(weight_mem_L3_data_in),
   .weight_mem_L1_data_out(weight_mem_L3_data_out),
   .weight_mem_L1_ena(weight_mem_L3_ena),
   .weight_mem_L2_wren(weight_mem_L4_wren),
   .weight_mem_L2_wr_addr(weight_mem_L4_wr_addr),
   .weight_mem_L2_data_in(weight_mem_L4_data_in),
   .weight_mem_L2_data_out(weight_mem_L4_data_out),
   .weight_mem_L2_ena(weight_mem_L4_ena)
    );  

// converge
wire [3:0] spike12;
//assign spike12 = i_spike_mem_wr_en[1]?i_spike_mem_dat:{s1[1],s2[1],s1[0],s2[0]};
assign spike12 = {s1[1],s2[1],s1[0],s2[0]};
wire valid12; 
assign valid12 = v1 && v2;
assign spike_out = spike12;
/////////////////////////////////////////////////
//    ____                  _                  //
//   / ___|___  _   _ _ __ | |_ ___ _ __ ___   //
//  | |   / _ \| | | | '_ \| __/ _ \ '__/ __|  //
//  | |__| (_) | |_| | | | | ||  __/ |  \__ \  //
//   \____\___/ \__,_|_| |_|\__\___|_|  |___/  //
//											   //
/////////////////////////////////////////////////                                          

// counters definition
reg [LAYERS_LOG2-1:0] layer_counter;
reg [clogb2(MAX_NEURONS/2-1)-1:0] neuron_cnt; 
reg [clogb2(MAX_SYNAPSES/4-1)-1:0] spike_wr_addr;

/////// SPIKE MEMORY WRITE PORT COUNTER /////////////////////////////////////////////////

reg spike_written;
reg [clogb2(LAYERS-1)-1:0] spike_written_counter;
wire [clogb2(MAX_SYNAPSES/4-1)-1:0] SYNAPSES;
`ifdef CONFIGURABILITY
	initial $display("Configurability is defined!");
	assign SYNAPSES = (spike_written_counter==0)?snn_input_channels/4-1:
					  (spike_written_counter==1)?neuron_1/4-1:
					  (spike_written_counter==2)?neuron_2/4-1:
										 neuron_3/4-1; //layer_counter==3
`else
	assign SYNAPSES = (spike_written_counter==0)?INPUT_SPIKE_1/4-1:
					  (spike_written_counter==1)?INPUT_SPIKE_2/4-1:
					  (spike_written_counter==2)?INPUT_SPIKE_3/4-1:
										 INPUT_SPIKE_4/4-1; //layer_counter==3
`endif
// it increases when: 
// 	1. spikes are received from the encoding slots (en is high)
//  2. spikes are produced from the SNN subprocessors layer_lp (v1|v2 is high)
wire spike_cnt_en;
assign spike_cnt_en = (en | v1 & layer_counter != LAYERS-1);
always @(posedge clk)
    if(rst)
        spike_wr_addr <= 0;
    else if (spike_cnt_en)
			if(spike_wr_addr == SYNAPSES)
        		spike_wr_addr <= 0;
			else
				spike_wr_addr <= spike_wr_addr + 1'b1;

// when the spike memory is loaded the inference starts
always @(posedge clk)
    if (rst)
        spike_written <= 0;
    else if( spike_cnt_en && (spike_wr_addr == SYNAPSES) )
            spike_written <= 1;
         else 
            spike_written <= 0;

always @(posedge clk)
    if (rst)
		spike_written_counter <= 0;
	else if(spike_written)
			spike_written_counter <= spike_written_counter +1'b1;

/////// NEURON COUNTER /////////////////////////////////////////////////

wire [clogb2(MAX_NEURONS/2-1)-1:0] NEURON;
`ifdef CONFIGURABILITY
	assign NEURON =   (layer_counter==0)? neuron_1/2-1:
					  (layer_counter==1)? neuron_2/2-1:
					  (layer_counter==2)? neuron_3/2-1:
										  neuron_4/2-1; //layer_counter==3
`else
	assign NEURON =   (layer_counter==0)? NEURON_1/2-1:
					  (layer_counter==1)? NEURON_2/2-1:
					  (layer_counter==2)? NEURON_3/2-1:
										  NEURON_4/2-1; //layer_counter==3
`endif

// neuron_cnt increases when the weights of a neuron are read
wire stream_out_done;
assign stream_out_done = stream_out_done_2 || stream_out_done_1;
always @(posedge clk)
    if(rst)
        neuron_cnt <= 0;
    else if (stream_out_done)
            if (neuron_cnt < NEURON)
                neuron_cnt <= neuron_cnt + 1'b1;
            else
                neuron_cnt <= 0;
                
assign layer_dispatched = stream_out_done && (neuron_cnt == NEURON); 

/////// LAYER COUNTER /////////////////////////////////////////////////
wire inference_done;
always @(posedge clk)
    if (rst)
        layer_counter <= 0;
    else if (layer_integrated)
			if(layer_counter < LAYERS)
        		layer_counter <= layer_counter+1'b1;

assign inference_done = (layer_counter == LAYERS) && layer_integrated;

//////////////////////////////////////////////////
//   _        _ __   _______ ____  ____         //
//  | |      / \\ \ / / ____|  _ \/ ___|        //
//  | |     / _ \\ V /|  _| | |_) \___ \        //
//  | |___ / ___ \| | | |___|  _ < ___) |       //
//  |_____/_/   \_\_| |_____|_| \_\____/        //
//    ____ ___  _   _ _____ ____   ___  _       //
//   / ___/ _ \| \ | |_   _|  _ \ / _ \| |      //
//  | |  | | | |  \| | | | | |_) | | | | |      //
//  | |__| |_| | |\  | | | |  _ <| |_| | |___   //
//   \____\___/|_| \_| |_| |_| \_\\___/|_____|  //
//   ____ ___ ____ _   _    _    _     ____     //
//  / ___|_ _/ ___| \ | |  / \  | |   / ___|    //
//  \___ \| | |  _|  \| | / _ \ | |   \___ \    //
//   ___) | | |_| | |\  |/ ___ \| |___ ___) |   //
//  |____/___\____|_| \_/_/   \_\_____|____/    //
//                                              //
//////////////////////////////////////////////////

/////// LAYER ENABLE /////////////////////////////////////////////////
reg layer_enable, layer_enable_d, layer_enable_dd;
always @(posedge clk)
    if (rst)
		layer_enable <= 0;
	else if (stream_out_1 | stream_out_2)
			layer_enable <= 1;
		 else if (stream_out_done && (neuron_cnt == NEURON))
				layer_enable <= 0;

always @(posedge clk)
    if (rst) begin
		layer_enable_d  <= 0;
		layer_enable_dd <= 0;
		end 
	else begin
		layer_enable_d  <= layer_enable;
		layer_enable_dd <= layer_enable_d;
		end

/////// LAYER ACCUMULATOR /////////////////////////////////////////////

wire convolution_valid;

// The conv module valid bit "convolution_pipe_full" is:  en -|_|-|_|-|_|-|_|-|_|- convolution_pipe_full
// By counting the number of clock cycles it is at logic-1, and comparing the value with 
// the number of memory reads per convolution (provided by the stack), it is possible to 
// generate the convolution valid signal "convolution_valid", which is used to: 
//  1. clear the accumulator register
//  2. enable the LIF neuron integration

assign words_to_read = layer_counter[0]? words_to_read_2 : words_to_read_1;

reg [clogb2(MAX_SYNAPSES/4-1)-1:0] convolution_valid_cnt;
always @(posedge clk)
    if (rst)
        convolution_valid_cnt <= 0;
    else if (convolution_pipe_full)
            if(convolution_valid_cnt < words_to_read)
                convolution_valid_cnt <= convolution_valid_cnt + 1'b1;
            else
                convolution_valid_cnt <= 0;
 
assign convolution_valid = (convolution_valid_cnt == words_to_read) && convolution_pipe_full; 

// Moreover, it is necessary to assess when each layer's computation terminates
// to clear the accumulation register

wire integrated_neuron_1, integrated_neuron_2;
assign integrated_neuron = integrated_neuron_1; // integrated_neuron 1 and 2 are the same

reg integrated_neuron_r;
always @(posedge clk)
    if(rst)
        integrated_neuron_r <= integrated_neuron;
    else 
        integrated_neuron_r <= integrated_neuron;

// Integrated neurons counter

reg [clogb2(MAX_NEURONS/2-1)-1:0] integrated_neurons_cnt, integrated_neurons_cnt_d; 
always @(posedge clk)
    if (rst)
        integrated_neurons_cnt <= 0;
    else if(integrated_neuron)
            if(integrated_neurons_cnt < NEURON)
                integrated_neurons_cnt <= integrated_neurons_cnt + 1'b1;
            else 
                integrated_neurons_cnt <= 0;

always @(posedge clk)
    if (rst)
        integrated_neurons_cnt_d <= 0;
    else
        integrated_neurons_cnt_d <= integrated_neurons_cnt;      
  
assign layer_integrated = integrated_neuron_r && (integrated_neurons_cnt_d == NEURON); 

//////////////////////////////
//      _             _     //
//  ___| |_ __ _  ___| | __ //
// / __| __/ _` |/ __| |/ / //
// \__ \ || (_| | (__|   <  //
// |___/\__\__,_|\___|_|\_\ //
//                          //
//////////////////////////////

// it stores the addresses of the active spike groups
// At every neuron integration:
//  - spike_rd_addr = stack_data_out
//  - weight_rd_addr = {neuron_cnt,stack_data_out}
// By doing so only the active groups are read, enabling
// both higher performances and least power wastes

wire stream_out_done_1, stream_out_done_2;
wire stream_out_1, stream_out_2;
wire [clogb2(MAX_SYNAPSES/4-1)-1:0] words_to_read_1, words_to_read_2, words_to_read;
wire empty, empty_1, empty_2;

wire stack_en_1;
assign stack_en_1 = (v1 && v2 && (ag1 || ag2) && layer_counter[0]) || (en & active_group_in);
assign stream_out_1 = (spike_written && ~spike_written_counter[0]) || (stream_out_done_1  && (neuron_cnt != NEURON));
stack
#(
.DATA_WIDTH(clogb2(MAX_SYNAPSES/4-1)),
.DEPTH(MAX_SYNAPSES/4)
 )
stack_1
 (
.clk(clk), .rst(rst),
.din(spike_wr_addr),
.wr_en(stack_en_1), 
.clear(layer_integrated && ~layer_counter[0]),
.stream_out(stream_out_1),

.dout(spike_rd_addr_1),
.done(stream_out_done_1),
.active_entries(words_to_read_1),
.empty(empty_1)
);

wire stack_en_2;
assign stack_en_2 = v1 && v2 && (ag1 || ag2) && ~layer_counter[0];
assign stream_out_2 = (spike_written && spike_written_counter[0]) || (stream_out_done_2 && (neuron_cnt != NEURON));
stack
#(
.DATA_WIDTH(clogb2(MAX_SYNAPSES/4-1)),
.DEPTH(MAX_SYNAPSES/4)
 )
stack_2
 (
.clk(clk), .rst(rst),
.din(spike_wr_addr),
.wr_en(stack_en_2), 
.clear(layer_integrated && layer_counter[0]),
.stream_out(stream_out_2),

.dout(spike_rd_addr_2),
.done(stream_out_done_2),
.active_entries(words_to_read_2),
.empty(empty_2)
);

// stack enable to stream out the rd_address for spike_mem and weight_mem
wire [clogb2(MAX_SYNAPSES/4-1)-1:0] spike_rd_addr;
assign spike_rd_addr = layer_counter[0]?spike_rd_addr_2:spike_rd_addr_1;
assign empty = layer_counter[0]?empty_2:empty_1;

/////////////////////////////////////////////////////  
//            _ _                                  //
//  ___ _ __ (_) | _____   _ __ ___   ___ _ __ ___ //
// / __| '_ \| | |/ / _ \ | '_ ` _ \ / _ \ '_ ` _  //
// \__ \ |_) | |   <  __/ | | | | | |  __/ | | | | //
// |___/ .__/|_|_|\_\___| |_| |_| |_|\___|_| |_| | //
//     |_|                                         //
///////////////////////////////////////////////////// 

wire [3:0] spike_mem_out_1;
wire [clogb2(MAX_SYNAPSES/4-1)-1:0] spike_rd_addr_1;
wire spike_wr_en_1; 

wire [3:0] spike_mem_out_2;
wire [clogb2(MAX_SYNAPSES/4-1)-1:0] spike_rd_addr_2;
wire spike_wr_en_2; 

//assign spike_wr_en_1 = (v1 && v2 &&  layer_counter[0]) || (en & active_group_in) || i_spike_mem_wr_en[0];
assign spike_wr_en_1 = (v1 && v2 &&  layer_counter[0]) || (en & active_group_in);
//assign spike_wr_en_2 = 	v1 && v2 && ~layer_counter[0] || i_spike_mem_wr_en[1]; 
assign spike_wr_en_2 = 	v1 && v2 && ~layer_counter[0];

wire [3:0] spike_mem_in_1;
//assign spike_mem_in_1 = i_spike_mem_wr_en[0]?i_spike_mem_dat:en?spike_in:spike12;
assign spike_mem_in_1 = en ? spike_in : spike12;

wire [clogb2(MAX_SYNAPSES/4-1)-1:0] spike_rd_addr_1_mux, spike_rd_addr_2_mux, spike_wr_addr_mux;
assign spike_rd_addr_1_mux = i_spike_mem_rd_en[0] ? i_spike_mem_adr : spike_rd_addr_1;
assign spike_rd_addr_2_mux = i_spike_mem_rd_en[1] ? i_spike_mem_adr : spike_rd_addr_2;
assign o_spike_mem_dat = {spike_mem_out_2,spike_mem_out_1};
//assign spike_wr_addr_mux = i_spike_mem_wr_en[0] | i_spike_mem_wr_en[1] ? i_spike_mem_adr : spike_wr_addr;
assign spike_wr_addr_mux = spike_wr_addr;


// written when en is 1 and layer is 0
BRAM_singlePort_readFirst #(
  .RAM_WIDTH(4),            
  .RAM_DEPTH(MAX_SYNAPSES/4),             
  .RAM_PERFORMANCE("HIGH_PERFORMANCE"), 
  .INIT_FILE("")          
)
	spike_mem_1
(
  .addra(spike_wr_addr_mux),    
  .addrb(spike_rd_addr_1_mux),  
  .dina(spike_mem_in_1),   
  .clk(clk),      
  .wea(spike_wr_en_1),      
  .ena(spike_wr_en_1),      
  .enb(1'b1),      
  .rst(rst),      
  .regceb(1'b1),
  
  .doutb(spike_mem_out_1)   
    );

BRAM_singlePort_readFirst #(
  .RAM_WIDTH(4),             
  .RAM_DEPTH(MAX_SYNAPSES/4),             
  .RAM_PERFORMANCE("HIGH_PERFORMANCE"), 
  .INIT_FILE("")             
)
	spike_mem_2
(
  .addra(spike_wr_addr_mux),   
  .addrb(spike_rd_addr_2_mux),  
  .dina(spike12),    
  .clk(clk),     
  .wea(spike_wr_en_2),  
  .ena(spike_wr_en_2), 
  .enb(1'b1),      
  .rst(rst),     
  .regceb(1'b1),
  
  .doutb(spike_mem_out_2)   // Port B RAM output data
    );

wire [3:0] spike_mem_out;
assign spike_mem_out = empty?4'b0:layer_counter[0]?spike_mem_out_2:spike_mem_out_1;

// weight_rd_addr concat neuron_cnt and spike_rd_addr 
wire [WEIGHT_ADDRESS_SIZE-1:0] weight_rd_addr;

assign weight_rd_addr = (layer_counter == 0)?{layer_counter, {P1{1'b0}},
											  neuron_cnt[clogb2(NEURON_1/2-1)-1:0], 
											  spike_rd_addr[SYN_G1_L2-1:0]}:
						(layer_counter == 1)?{layer_counter, {P2{1'b0}},
											  neuron_cnt[clogb2(NEURON_2/2-1)-1:0], 
											  spike_rd_addr[SYN_G2_L2-1:0]}:
						(layer_counter == 2)?{layer_counter, {P3{1'b0}},
											  neuron_cnt[clogb2(NEURON_3/2-1)-1:0], 
											  spike_rd_addr[SYN_G3_L2-1:0]}:
	      									 {layer_counter, {P4{1'b0}},
											  neuron_cnt[clogb2(NEURON_4/2-1)-1:0], 
											  spike_rd_addr[SYN_G4_L2-1:0]};
///////////////////////////////////////////
//   ___  _   _ _____ ____  _   _ _____  //
//  / _ \| | | |_   _|  _ \| | | |_   _| //
// | | | | | | | | | | |_) | | | | | |   //
// | |_| | |_| | | | |  __/| |_| | | |   //
//  \___/ \___/  |_| |_|    \___/  |_|   //
//  ____  _   _ _____ _____ _____ ____   //
// | __ )| | | |  ___|  ___| ____|  _ \  //
// |  _ \| | | | |_  | |_  |  _| | |_) | //
// | |_) | |_| |  _| |  _| | |___|  _ <  //
// |____/ \___/|_|   |_|   |_____|_| \_\ //
//                                       //
///////////////////////////////////////////

wire output_buffer_wr_en;
wire [31:0] output_buffer_din;
assign output_buffer_wr_en = integrated_neuron && layer_counter == 3;
assign output_buffer_din = {voltage_2,voltage_1};

BRAM_singlePort_readFirst #(
  .RAM_WIDTH(2*WIDTH),            
  .RAM_DEPTH(NEURON_4/2),             
  .RAM_PERFORMANCE("LOW_LATENCY"), 
  .INIT_FILE("")          
)
	output_buffer
(
  .addra(integrated_neurons_cnt),   //  
  .addrb(output_buffer_addr),  
  .dina(output_buffer_din),   
  .clk(clk),      
  .wea(output_buffer_wr_en),      
  .ena(output_buffer_wr_en),      
  .enb(output_buffer_ren),      
  .rst(rst),      
  .regceb(1'b1),
  
  .doutb(output_buffer_out)   //
    );

`ifdef CONFIGURABILITY	
	assign valid = integrated_neuron_r && integrated_neurons_cnt_d == NEURON && layer_counter == layers-1;	
	assign valid_spike = valid12 && layer_counter == layers-1;
`else 
	assign valid = integrated_neuron_r && integrated_neurons_cnt_d == NEURON && layer_counter == LAYERS-1;	
	assign valid_spike = valid12 && layer_counter == LAYERS-1;
`endif
	

//  The following function calculates the address width based on specified RAM depth
	function integer clogb2;
	  input integer depth;
		for (clogb2=0; depth>0; clogb2=clogb2+1)
		  depth = depth >> 1;
	endfunction 
	
	function integer max;
	  input integer j,k;
		if(j>k)
		  max = j;
	endfunction 	

	function integer max4;
        input integer a, b, c, d;
        begin
            max4 = (a > b ? a : b) > (c > d ? c : d) ? (a > b ? a : b) : (c > d ? c : d);
        end
    endfunction

endmodule

`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 04.11.2022 15:31:20
// Design Name: 
// Module Name: s2p
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module s2p #(parameter P = 2)(
    input clk, rst, en, spike_s,
    output reg [P-1:0] spike_p,
    output reg valid, 
    output reg active_group 
    );
    
    // clear logic
    wire end_cnt;
    always @(posedge clk)
        if (rst)
            valid <= 0;
        else 
            valid <= end_cnt;
     
    reg [clogb2(P-1)-1:0] cnt;
    always @(posedge clk)
        if (rst)
            cnt <= 0;
        else if(en)
            cnt <= cnt + 1'b1;
            
     assign end_cnt = (cnt == P-1) & en;
    
    
    // Serial 2 Parallel
	integer i;
    always @(posedge clk)
        if (rst)
            spike_p <= 0;
        else if(en) begin
            spike_p[0] <= spike_s;
            for(i=1;i<P;i=i+1)
				spike_p[i] <= spike_p[i-1];
            end
                else if(valid)
                    spike_p <= 0;
    
    // check if is present at least an input equal to 1
    always @(posedge clk)
        if (rst)
            active_group <= 0;
        else if(en) 
            active_group <= (active_group && ~valid) | spike_s;
             else if(valid)
                    active_group <= 0;
    

//  The following function calculates the address width based on specified RAM depth
function integer clogb2;
  input integer depth;
    for (clogb2=0; depth>0; clogb2=clogb2+1)
      depth = depth >> 1;

endfunction      
     
endmodule

`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03.11.2022 11:06:53
// Design Name: 
// Module Name: stack
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module stack
#(
parameter DATA_WIDTH = 4,
parameter DEPTH = 24
)
(
input clk, rst,
input [DATA_WIDTH-1:0] din,
input wr_en, clear,
input stream_out,

output [DATA_WIDTH-1:0] dout,
output reg done,
output [clogb2(DEPTH-1)-1:0] active_entries,
output empty 
);

// shift register    
reg [DATA_WIDTH-1:0] shift [DEPTH-1:0];
integer i;
always@(posedge clk)
    if (rst)
       for(i=0;i<DEPTH;i=i+1)
            shift[i] <= 0;
    else    
        if(wr_en) begin
            shift[0] <= din;
            for(i=1;i<DEPTH;i=i+1)
                shift[i] <= shift[i-1];
        end
            
// entries counter
reg [clogb2(DEPTH-1):0] entries_cnt;    
always @(posedge clk)
    if(rst)
       entries_cnt <= 0;
    else
        if(wr_en)
            entries_cnt <= entries_cnt + 1'b1;     
        else if (clear)
            entries_cnt <= 0;

// stream counter
reg [clogb2(DEPTH-1)-1:0] stream_cnt;    
always @(posedge clk)
    if(rst)
       stream_cnt <= 0;
    else
        if(stream_out && (entries_cnt != 0) )
            stream_cnt <= entries_cnt - 1'b1;
        else if (stream_cnt != 0) 
            stream_cnt <= stream_cnt - 1'b1;

// output assignment 
assign dout = shift[stream_cnt];

always @(posedge clk)
    if (rst) 
        done <= 0;
    else if (stream_cnt == 1 || ( stream_out && ( (entries_cnt == 1) || (entries_cnt == 0) ) ) )
            done <= 1;
         else
            done <= 0;

assign active_entries = entries_cnt == 0 ? 0 : entries_cnt - 1'b1;            
 
assign empty = entries_cnt == 0;
           
//  The following function calculates the address width based on specified RAM depth
function integer clogb2;
  input integer depth;
    for (clogb2=0; depth>0; clogb2=clogb2+1)
      depth = depth >> 1;
endfunction 
   
endmodule

`undef BRAM
