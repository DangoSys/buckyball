package examples.balls.common

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

class ShiftAddMul(w: Int) extends Module {
  require(w >= 1, "ShiftAddMul width must be >= 1")

  val io = IO(new Bundle {
    val start = Input(Bool())
    val a     = Input(UInt(w.W))
    val b     = Input(UInt(w.W))
    val p     = Output(UInt((2 * w).W))
    val done  = Output(Bool())
  })

  val acc    = Reg(UInt((2 * w).W))
  val sumS   = Reg(UInt((2 * w).W))
  val sumC   = Reg(UInt((2 * w).W))
  val loReg  = Reg(UInt((w + 1).W))
  val mcand  = Reg(UInt((2 * w).W))
  val mplier = Reg(UInt(w.W))
  val aCap   = Reg(UInt(w.W))
  val bCap   = Reg(UInt(w.W))
  val cnt    = Reg(UInt(log2Ceil(w + 1).W))
  val load   = RegInit(false.B)
  val busy   = RegInit(false.B)
  val cpaLo  = RegInit(false.B)
  val cpaHi  = RegInit(false.B)
  val done   = RegInit(false.B)

  io.p    := acc
  io.done := done

  done := false.B
  when(io.start) {
    aCap  := io.a
    bCap  := io.b
    load  := true.B
    busy  := false.B
    cpaLo := false.B
    cpaHi := false.B
  }.elsewhen(load) {
    sumS   := 0.U
    sumC   := 0.U
    mcand  := Cat(0.U(w.W), aCap)
    mplier := bCap
    cnt    := w.U
    load   := false.B
    busy   := true.B
  }.elsewhen(busy) {
    val addend = Mux(mplier(0), mcand, 0.U((2 * w).W))
    val ps     = sumS ^ sumC ^ addend
    val pc     = (sumS & sumC) | (sumS & addend) | (sumC & addend)
    sumS   := ps
    sumC   := Cat(pc(2 * w - 2, 0), 0.U(1.W))
    mcand  := Cat(mcand(2 * w - 2, 0), 0.U(1.W))
    mplier := mplier >> 1
    cnt    := cnt - 1.U
    when(cnt === 1.U) {
      busy  := false.B
      cpaLo := true.B
    }
  }.elsewhen(cpaLo) {
    loReg := sumS(w - 1, 0) +& sumC(w - 1, 0)
    cpaLo := false.B
    cpaHi := true.B
  }.elsewhen(cpaHi) {
    val hi = (sumS(2 * w - 1, w) +& sumC(2 * w - 1, w)) +& loReg(w)
    assert(hi(w) === 0.U, "ShiftAddMul CPA overflow")
    acc   := Cat(hi(w - 1, 0), loReg(w - 1, 0))
    cpaHi := false.B
    done  := true.B
  }
}

class Fp32Mul extends Module {

  val io = IO(new Bundle {
    val start  = Input(Bool())
    val a      = Input(UInt(32.W))
    val b      = Input(UInt(32.W))
    val result = Output(UInt(32.W))
    val done   = Output(Bool())
  })

  val umul    = Module(new ShiftAddMul(24))
  val aReg    = Reg(UInt(32.W))
  val bReg    = Reg(UInt(32.W))
  val result  = Reg(UInt(32.W))
  val doneReg = RegInit(false.B)
  val waitU   = RegInit(false.B)
  val pending = RegInit(false.B)

  io.result     := result
  io.done       := doneReg
  umul.io.start := false.B
  umul.io.a     := 0.U
  umul.io.b     := 0.U
  doneReg       := false.B

  when(io.start) {
    aReg := io.a
    bReg := io.b
    when((io.a(30, 0) === 0.U) || (io.b(30, 0) === 0.U)) {
      result  := 0.U
      doneReg := true.B
      waitU   := false.B
      pending := false.B
    }.otherwise {
      pending := true.B
      waitU   := true.B
    }
  }.elsewhen(pending) {
    pending       := false.B
    umul.io.start := true.B
    umul.io.a     := Cat(1.U(1.W), aReg(22, 0))
    umul.io.b     := Cat(1.U(1.W), bReg(22, 0))
  }.elsewhen(waitU && umul.io.done) {
    waitU   := false.B
    doneReg := true.B
    val aSign      = aReg(31)
    val bSign      = bReg(31)
    val aExp       = aReg(30, 23)
    val bExp       = bReg(30, 23)
    val prod       = umul.io.p
    val sig        = Wire(UInt(24.W))
    val guard      = Wire(Bool())
    val round      = Wire(Bool())
    val sticky     = Wire(Bool())
    val normAdjust = Wire(UInt(2.W))
    when(prod(47)) {
      sig        := prod(47, 24)
      guard      := prod(23)
      round      := prod(22)
      sticky     := prod(21, 0).orR
      normAdjust := 1.U
    }.otherwise {
      sig        := prod(46, 23)
      guard      := prod(22)
      round      := prod(21)
      sticky     := prod(20, 0).orR
      normAdjust := 0.U
    }
    val rounded    = sig +& (guard && (round || sticky || sig(0))).asUInt
    val exp        = aExp +& bExp +& normAdjust +& rounded(24) - 127.U
    when(exp(9)) {
      result := 0.U
    }.elsewhen(exp(8)) {
      result := Cat(aSign ^ bSign, 255.U(8.W), 0.U(23.W))
    }.otherwise {
      result := Cat(aSign ^ bSign, exp(7, 0), Mux(rounded(24), rounded(23, 1), rounded(22, 0)))
    }
  }
}
