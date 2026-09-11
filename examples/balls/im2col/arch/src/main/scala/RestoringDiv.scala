package examples.balls.im2col

import chisel3._
import chisel3.util._

class RestoringDiv(n: Int, d: Int) extends Module {
  require(n >= 1 && d >= 1, "RestoringDiv widths must be >= 1")

  val io = IO(new Bundle {
    val start = Input(Bool())
    val a     = Input(UInt(n.W))
    val b     = Input(UInt(d.W))
    val q     = Output(UInt(n.W))
    val r     = Output(UInt(d.W))
    val busy  = Output(Bool())
    val done  = Output(Bool())
  })

  val rem     = Reg(UInt((d + 1).W))
  val quot    = Reg(UInt(n.W))
  val divisor = Reg(UInt(d.W))
  val cnt     = Reg(UInt(log2Ceil(n + 1).W))
  val busyReg = RegInit(false.B)
  val doneReg = RegInit(false.B)

  io.q    := quot
  io.r    := rem(d - 1, 0)
  io.busy := busyReg
  io.done := doneReg

  doneReg := false.B
  when(io.start) {
    rem     := 0.U
    quot    := io.a
    divisor := io.b
    cnt     := n.U
    busyReg := true.B
  }.elsewhen(busyReg) {
    val shifted = Cat(rem(d - 1, 0), quot(n - 1))
    val ge      = shifted >= divisor
    rem  := Mux(ge, shifted - divisor, shifted)
    quot := Cat(quot(n - 2, 0), ge.asUInt)
    cnt  := cnt - 1.U
    when(cnt === 1.U) {
      busyReg := false.B
      doneReg := true.B
    }
  }
}
