package examples.balls.mxmm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}

@instantiable
class Panels(bankEntries: Int) extends Module {

  @public val io = IO(new Bundle {
    val write       = Input(Vec(2, Bool()))
    val scaleWrite  = Input(Vec(2, Bool()))
    val lane        = Input(Vec(2, UInt(4.W)))
    val address     = Input(Vec(2, UInt(log2Ceil(bankEntries).W)))
    val word        = Input(Vec(2, UInt(128.W)))
    val scale       = Input(Vec(2, UInt(8.W)))
    val read        = Input(Bool())
    val rowGroup    = Input(UInt(12.W))
    val columnGroup = Input(UInt(12.W))
    val k           = Input(UInt(34.W))
    val reduction   = Input(UInt(34.W))
    val mxfp8       = Input(Bool())
    val vector      = Input(Bool())
    val a           = Output(Vec(16, UInt(32.W)))
    val b           = Output(Vec(16, UInt(32.W)))
    val valid       = Output(Bool())
  })

  val mxfp8  = RegNext(io.mxfp8)
  val vector = RegNext(io.vector)
  val offset = RegNext(Mux(io.mxfp8, io.k(3, 0) << 3, io.k(1, 0) << 5))
  io.valid := RegNext(io.read, false.B)
  for {
    operand <- 0 until 2
    lane    <- 0 until 16
  } {
    val depth        = bankEntries / 16
    val codes        = SyncReadMem(depth, UInt(128.W))
    val scales       = SyncReadMem(depth / 2, UInt(8.W))
    when(io.write(operand) && io.lane(operand) === lane.U) {
      assert(io.address(operand) < depth.U)
      codes.write(io.address(operand), io.word(operand))
    }
    when(io.scaleWrite(operand) && io.lane(operand) === lane.U) {
      assert(io.address(operand) < (depth / 2).U)
      scales.write(io.address(operand), io.scale(operand))
    }
    val group        = if (operand == 0) io.rowGroup else io.columnGroup
    val rowWords     = Mux(io.mxfp8, io.reduction >> 4, io.reduction >> 2)
    val codeAddress  = group * rowWords + Mux(io.mxfp8, io.k >> 4, io.k >> 2)
    val scaleAddress = group * (io.reduction >> 5) + (io.k >> 5)
    val active       = if (operand == 0 && lane > 0) !io.vector else true.B
    val word         = codes.read(codeAddress, io.read && active)
    val scale        = scales.read(scaleAddress, io.read && active && io.mxfp8)
    val decode       = Instantiate(new Mxfp8Decode)
    decode.io.code  := (word >> offset)(7, 0)
    decode.io.scale := scale
    val value   = Mux(mxfp8, decode.io.out, (word >> offset)(31, 0))
    val enabled = if (operand == 0 && lane > 0) !vector else true.B
    when(io.valid && enabled) {
      assert(Mux(mxfp8, !decode.io.invalid, value(30, 23) =/= 255.U), "MATMUL operands must decode to finite FP32")
    }
    if (operand == 0) io.a(lane) := Mux(enabled, value, 0.U)
    else io.b(lane)              := value
  }
}
