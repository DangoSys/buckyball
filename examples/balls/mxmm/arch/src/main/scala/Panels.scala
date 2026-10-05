package examples.balls.mxmm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}

@instantiable
class Panels(bankEntries: Int) extends Module {

  @public
  val io = IO(new Bundle {
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
    val codeWrite    = io.write(operand) && io.lane(operand) === lane.U
    val scaleWrite   = io.scaleWrite(operand) && io.lane(operand) === lane.U
    when(codeWrite)(assert(io.address(operand) < depth.U))
    when(scaleWrite)(assert(io.address(operand) < (depth / 2).U))
    val group        = if (operand == 0) io.rowGroup else io.columnGroup
    val rowWords     = Mux(io.mxfp8, io.reduction >> 4, io.reduction >> 2)
    val codeAddress  = group * rowWords + Mux(io.mxfp8, io.k >> 4, io.k >> 2)
    val scaleAddress = group * (io.reduction >> 5) + (io.k >> 5)
    val active       = if (operand == 0 && lane > 0) !io.vector else true.B
    val codeRead     = io.read && active
    val scaleRead    = codeRead && io.mxfp8
    assert(!(codeRead && codeWrite))
    assert(!(scaleRead && scaleWrite))
    val word         = codes.readWrite(
      Mux(codeWrite, io.address(operand), codeAddress),
      io.word(operand),
      codeRead || codeWrite,
      codeWrite
    )
    val scale        = scales.readWrite(
      Mux(scaleWrite, io.address(operand), scaleAddress),
      io.scale(operand),
      scaleRead || scaleWrite,
      scaleWrite
    )
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
