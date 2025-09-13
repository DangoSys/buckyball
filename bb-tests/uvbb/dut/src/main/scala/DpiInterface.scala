package uvbb

import chisel3._
import chisel3.util._
import chisel3.experimental._

// SRAM DPI-C接口
class SramDpiC(addrWidth: Int, elemWidth: Int, elemNum: Int) extends BlackBox(Map(
  "ADDR_WIDTH" -> IntParam(addrWidth),
  "ELEM_WIDTH" -> IntParam(elemWidth),
  "ELEM_NUM"   -> IntParam(elemNum)
)) with HasBlackBoxResource {
  val dataWidth = elemWidth * elemNum
  val io = IO(new Bundle {
    val addr = Input(UInt(addrWidth.W))
    val rdata = Output(UInt(dataWidth.W))
    val wdata = Input(UInt(dataWidth.W))
    val ren = Input(Bool())
    val wen = Input(Bool())
    val mask = Input(UInt(elemNum.W))
  })
  addResource("dpic.sv")
}

// ACC DPI-C接口
class AccDpiC(addrWidth: Int, elemWidth: Int, elemNum: Int, maskLen: Int) extends BlackBox(Map(
  "ADDR_WIDTH" -> IntParam(addrWidth),
  "ELEM_WIDTH" -> IntParam(elemWidth),
  "ELEM_NUM"   -> IntParam(elemNum),
  "MASK_LEN"   -> IntParam(maskLen)
)) with HasBlackBoxResource {
  val dataWidth = elemWidth * elemNum
  val io = IO(new Bundle {
    val addr = Input(UInt(addrWidth.W))
    val rdata = Output(UInt(dataWidth.W))
    val wdata = Input(UInt(dataWidth.W))
    val ren = Input(Bool())
    val wen = Input(Bool())
    val mask = Input(UInt(maskLen.W))
  })
  addResource("dpic.sv")
}

// 命令DPI-C接口
class CmdDpiC extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle {
    val cmdReq_valid := Input(Bool())
    val cmdReq_ready := Input(Bool())
    val cmdReq_bits_cmd_bid := Input(UInt(4.W))
    val cmdReq_bits_cmd_iter := Input(UInt(10.W))
    val cmdReq_bits_cmd_special := Input(UInt(40.W))
    val cmdReq_bits_rob_id := Input(UInt(8.W))
    val cmdResp_valid := Input(Bool())
    val cmdResp_ready := Input(Bool())
    val cmdResp_bits_rob_id := Input(UInt(8.W))
  })
  addResource("dpic.sv")
}
