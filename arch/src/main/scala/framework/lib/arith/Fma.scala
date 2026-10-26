// See LICENSE.Berkeley and LICENSE.SiFive for license details.
package framework.lib.arith

import chisel3.experimental.hierarchy.{Instance, Instantiate}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

@instantiable
class Fma(latency: Int, expWidth: Int, sigWidth: Int) extends Module {
  override def desiredName = s"Fma_l${latency}_e${expWidth}_s$sigWidth"
  require(latency <= 4)

  @public
  val io = IO(new Bundle {
    val validin        = Input(Bool())
    val op             = Input(Bits(2.W))
    val a              = Input(Bits((expWidth + sigWidth + 1).W))
    val b              = Input(Bits((expWidth + sigWidth + 1).W))
    val c              = Input(Bits((expWidth + sigWidth + 1).W))
    val roundingMode   = Input(UInt(3.W))
    val detectTininess = Input(UInt(1.W))
    val out            = Output(Bits((expWidth + sigWidth + 1).W))
    val exceptionFlags = Output(Bits(5.W))
    val validout       = Output(Bool())
  })

  //------------------------------------------------------------------------
  //------------------------------------------------------------------------

  val mulAddRecFNToRaw_preMul  = Instantiate(new hardfloat.MulAddRecFNToRaw_preMul(expWidth, sigWidth))
  val mulAddRecFNToRaw_postMul = Instantiate(new hardfloat.MulAddRecFNToRaw_postMul(expWidth, sigWidth))

  mulAddRecFNToRaw_preMul.io.op := io.op
  mulAddRecFNToRaw_preMul.io.a  := io.a
  mulAddRecFNToRaw_preMul.io.b  := io.b
  mulAddRecFNToRaw_preMul.io.c  := io.c

  // Isolate preMul's unpack/alignment logic from the multiplier-add stage.
  val preMulRegs      = if (latency >= 3) 1 else 0
  val preMulMulAddA   = Pipe(io.validin, mulAddRecFNToRaw_preMul.io.mulAddA, preMulRegs)
  val preMulMulAddB   = Pipe(io.validin, mulAddRecFNToRaw_preMul.io.mulAddB, preMulRegs)
  val preMulMulAddC   = Pipe(io.validin, mulAddRecFNToRaw_preMul.io.mulAddC, preMulRegs)
  val preMulToPostMul = Pipe(io.validin, mulAddRecFNToRaw_preMul.io.toPostMul, preMulRegs)

  // Keep the full-precision product unrounded while separating multiplication
  // from the final addition.  The aligned addend and post-processing metadata
  // must cross this stage with the product to preserve fused FMA semantics.
  val mulRegs       = if (latency == 4) 1 else 0
  val mulResult     = preMulMulAddA.bits * preMulMulAddB.bits
  val mulResultPipe = Pipe(preMulToPostMul.valid, mulResult, mulRegs)
  val mulAddCPipe   = Pipe(preMulToPostMul.valid, preMulMulAddC.bits, mulRegs)
  val mulToPostMul  = Pipe(preMulToPostMul.valid, preMulToPostMul.bits, mulRegs)
  val mulAddResult  = mulResultPipe.bits +& mulAddCPipe.bits

  val valid_stage0          = Wire(Bool())
  val roundingMode_stage0   = Wire(UInt(3.W))
  val detectTininess_stage0 = Wire(UInt(1.W))

  val postmul_regs = if (latency > 0) 1 else 0
  mulAddRecFNToRaw_postMul.io.fromPreMul   := Pipe(mulToPostMul.valid, mulToPostMul.bits, postmul_regs).bits
  mulAddRecFNToRaw_postMul.io.mulAddResult := Pipe(mulToPostMul.valid, mulAddResult, postmul_regs).bits
  mulAddRecFNToRaw_postMul.io.roundingMode := Pipe(
    io.validin,
    io.roundingMode,
    preMulRegs + mulRegs + postmul_regs
  ).bits
  roundingMode_stage0                      := Pipe(io.validin, io.roundingMode, preMulRegs + mulRegs + postmul_regs).bits
  detectTininess_stage0                    := Pipe(io.validin, io.detectTininess, preMulRegs + mulRegs + postmul_regs).bits
  valid_stage0                             := Pipe(io.validin, false.B, preMulRegs + mulRegs + postmul_regs).valid

  //------------------------------------------------------------------------
  //------------------------------------------------------------------------

  val roundRawFNToRecFN = Instantiate(new hardfloat.RoundRawFNToRecFN(expWidth, sigWidth, 0))

  val round_regs = if (latency >= 2) 1 else 0
  roundRawFNToRecFN.io.invalidExc     := Pipe(valid_stage0, mulAddRecFNToRaw_postMul.io.invalidExc, round_regs).bits
  roundRawFNToRecFN.io.in             := Pipe(valid_stage0, mulAddRecFNToRaw_postMul.io.rawOut, round_regs).bits
  roundRawFNToRecFN.io.roundingMode   := Pipe(valid_stage0, roundingMode_stage0, round_regs).bits
  roundRawFNToRecFN.io.detectTininess := Pipe(valid_stage0, detectTininess_stage0, round_regs).bits
  io.validout                         := Pipe(valid_stage0, false.B, round_regs).valid

  roundRawFNToRecFN.io.infiniteExc := false.B

  io.out            := roundRawFNToRecFN.io.out
  io.exceptionFlags := roundRawFNToRecFN.io.exceptionFlags
}
