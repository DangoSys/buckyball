// -- SRAM DPI-C Interface -- //
// SRAM: 8bit × 16个 = 128位，字节寻址
import "DPI-C" function void sram_read(
  input int bank_id,
  input int addr,
  output byte data [0:15]  // 16个字节数组
);

import "DPI-C" function void sram_write(
  input int bank_id,
  input int addr,
  input byte data [0:15],   // 16个字节数组
  input int mask
);

module SramDpiC #(
  parameter ADDR_WIDTH = 12,
  parameter ELEM_WIDTH = 8,   // 字节宽度
  parameter ELEM_NUM   = 16   // 16个字节
)(
  input  logic [ADDR_WIDTH-1:0] addr,
  output logic [ELEM_WIDTH*ELEM_NUM-1:0] rdata,
  input  logic [ELEM_WIDTH*ELEM_NUM-1:0] wdata,
  input  logic ren,
  input  logic wen,
  input  logic [ELEM_NUM-1:0] mask
);
  int bank_id = 0;

  always_comb begin
    byte read_data [0:15];
    if (ren) begin
      sram_read(bank_id, int'(addr), read_data);
      // 数组拼接成向量，字节0在最低位
      for (int i = 0; i < 16; i++) begin
        rdata[i*8 +: 8] = read_data[i];
      end
    end else begin
      rdata = '0;
    end
  end

  always @(posedge wen) begin
    if (wen) begin
      byte write_data [0:15];
      // 向量拆成数组
      for (int i = 0; i < 16; i++) begin
        write_data[i] = wdata[i*8 +: 8];
      end
      sram_write(bank_id, int'(addr), write_data, int'(mask));
    end
  end
endmodule

// -- ACC DPI-C Interface -- //
// ACC: 32bit × 4个 = 128位，字寻址
import "DPI-C" function void acc_read(
  input int bank_id,
  input int addr,
  output int data [0:3]  // 4个字数组
);

import "DPI-C" function void acc_write(
  input int bank_id,
  input int addr,
  input int data [0:3],   // 4个字数组
  input int mask
);

module AccDpiC #(
  parameter ADDR_WIDTH = 12,
  parameter ELEM_WIDTH = 32,  // 字宽度
  parameter ELEM_NUM   = 4,   // 4个字
  parameter MASK_LEN   = 16   // mask位宽
)(
  input  logic [ADDR_WIDTH-1:0] addr,
  output logic [ELEM_WIDTH*ELEM_NUM-1:0] rdata,
  input  logic [ELEM_WIDTH*ELEM_NUM-1:0] wdata,
  input  logic ren,
  input  logic wen,
  input  logic [MASK_LEN-1:0] mask
);
  int bank_id = 0;

  always_comb begin
    int read_data [0:3];
    if (ren) begin
      acc_read(bank_id, int'(addr), read_data);
      // 数组拼接成向量，字0在最低位
      for (int i = 0; i < 4; i++) begin
        rdata[i*32 +: 32] = read_data[i];
      end
    end else begin
      rdata = '0;
    end
  end

  always @(posedge wen) begin
    if (wen) begin
      int write_data [0:3];
      // 向量拆成数组
      for (int i = 0; i < 4; i++) begin
        write_data[i] = wdata[i*32 +: 32];
      end
      acc_write(bank_id, int'(addr), write_data, int'(mask));
    end
  end
endmodule

// -- CMD DPI-C Interface -- //
import "DPI-C" function void cmd_request(
  input bit valid,
  input bit ready,
  input int bid,
  input int iter,
  input longint special,
  input int rob_id
);

import "DPI-C" function void cmd_response(
  input bit valid,
  input bit ready,
  input int rob_id
);

module CmdDpiC (
  input  logic cmdReq_valid,
  input  logic cmdReq_ready,
  input  logic [3:0] cmdReq_bits_cmd_bid,
  input  logic [9:0] cmdReq_bits_cmd_iter,
  input  logic [39:0] cmdReq_bits_cmd_special,
  input  logic [7:0] cmdReq_bits_rob_id,
  input  logic cmdResp_valid,
  input  logic cmdResp_ready,
  input  logic [7:0] cmdResp_bits_rob_id
);
  always @(posedge cmdReq_valid) begin
    if (cmdReq_valid && cmdReq_ready) begin
      cmd_request(cmdReq_valid, cmdReq_ready, int'(cmdReq_bits_cmd_bid),
                 int'(cmdReq_bits_cmd_iter), longint'(cmdReq_bits_cmd_special),
                 int'(cmdReq_bits_rob_id));
    end
  end

  always @(posedge cmdResp_valid) begin
    if (cmdResp_valid && cmdResp_ready) begin
      cmd_response(cmdResp_valid, cmdResp_ready, int'(cmdResp_bits_rob_id));
    end
  end
endmodule
