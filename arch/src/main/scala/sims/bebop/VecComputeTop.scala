package sims.bebop

import chisel3._
import chisel3.util._
import framework.balldomain.prototype.vector.configs.VectorBallParam
import framework.balldomain.prototype.vector.op.MulOp

class VecComputeTop extends Module {
  private val cfg = VectorBallParam()
  require(cfg.lane == 16, s"VecComputeTop requires lane=16, got ${cfg.lane}")
  require(cfg.inputWidth == 8, s"VecComputeTop requires inputWidth=8, got ${cfg.inputWidth}")
  require(cfg.outputWidth == 32, s"VecComputeTop requires outputWidth=32, got ${cfg.outputWidth}")

  val io = IO(new Bundle {
    val start = Input(Bool())
    val iter  = Input(UInt(16.W))
    val op1   = Input(Vec(16, UInt(8.W)))
    val op2   = Input(Vec(16, UInt(8.W)))
    val res   = Output(Vec(16, UInt(32.W)))
    val valid = Output(Bool())
    val done  = Output(Bool())
  })

  val mul = Module(new MulOp(cfg) {
    override def desiredName = "MulOpVecComputeTop"
  })

  val op1Reg = Reg(Vec(16, UInt(8.W)))
  val op2Reg = Reg(Vec(16, UInt(8.W)))
  val inFire = RegInit(false.B)

  val rowCnt = RegInit(0.U(4.W))
  val active = RegInit(false.B)
  val doneR  = RegInit(false.B)

  when(io.start) {
    assert(io.iter =/= 0.U, "VecComputeTop: iter must be non-zero")
    op1Reg := io.op1
    op2Reg := io.op2
    inFire := true.B
    rowCnt := 0.U
    active := true.B
    doneR  := false.B
  }.otherwise {
    inFire := false.B
    when(active && mul.io.out.valid) {
      when(rowCnt === 15.U) {
        active := false.B
        doneR  := true.B
      }.otherwise {
        rowCnt := rowCnt + 1.U
      }
    }.elsewhen(doneR) {
      doneR := false.B
    }
  }

  mul.io.in.valid    := inFire
  mul.io.in.bits.in1 := op1Reg
  mul.io.in.bits.in2 := op2Reg
  mul.io.out.ready   := true.B

  io.res   := mul.io.out.bits.out
  io.valid := active && mul.io.out.valid
  io.done  := doneR
}
