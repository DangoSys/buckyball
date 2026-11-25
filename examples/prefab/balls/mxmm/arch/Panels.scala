package examples.balls.mxmm

import examples.balls.mxmm.configs.MxmmBallParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}

@instantiable
class Panels(val bankEntries: Int, val ballParam: MxmmBallParam) extends Module {

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
    val vector      = Input(Bool())
    val a           = Output(Vec(16, new Mxfp8Value))
    val b           = Output(Vec(16, new Mxfp8Value))
    val valid       = Output(Bool())
  })

  val vector = RegNext(io.vector)
  val offset = RegNext(io.k(3, 0) << 3)
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
    val rowWords     = io.reduction >> 4
    val codeAddress  = group * rowWords + (io.k >> 4)
    val scaleAddress = group * (io.reduction >> 5) + (io.k >> 5)
    val active       = if (operand == 0 && lane > 0) !io.vector else true.B
    val codeRead     = io.read && active
    val scaleRead    = codeRead
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
    val value   = decode.io.out
    val enabled = if (operand == 0 && lane > 0) !vector else true.B
    when(io.valid && enabled) {
      assert(!decode.io.invalid, "MATMUL operands must decode to finite FP32")
    }
    if (operand == 0) io.a(lane) := Mux(enabled, value, 0.U.asTypeOf(new Mxfp8Value))
    else io.b(lane)              := value
  }
}
