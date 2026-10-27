package examples.balls.int8add

import chisel3.experimental.hierarchy.{instantiable, public}

import chisel3._
import chisel3.util._
import hardfloat.{countLeadingZeros, isSigNaNRawFloat, lowMask, orReduceBy2, orReduceBy4, RawFloat}
import hardfloat.consts.round_min

/**
 * Throughput-oriented raw floating-point adder used by Int8AddBall.
 *
 * The Berkeley HardFloat AddRawFN implementation is combinational.  This
 * wrapper keeps the same raw-float equations but registers the alignment,
 * normalization, and final select boundaries.  It accepts one operand pair
 * per cycle and reports the result four cycles later.
 */
@instantiable
class PipelinedAddRawFN(val expWidth: Int, val sigWidth: Int) extends Module {

  @public
  val io = IO(new Bundle {
    val validin      = Input(Bool())
    val subOp        = Input(Bool())
    val a            = Input(new RawFloat(expWidth, sigWidth))
    val b            = Input(new RawFloat(expWidth, sigWidth))
    val roundingMode = Input(UInt(3.W))
    val validout     = Output(Bool())
    val invalidExc   = Output(Bool())
    val rawOut       = Output(new RawFloat(expWidth, sigWidth + 2))
  })

  val alignDistWidth = log2Ceil(sigWidth)

  // Stage 0: register the raw operands and exponent comparison.
  val a0            = RegNext(io.a)
  val b0            = RegNext(io.b)
  val effSignB0     = RegNext(io.b.sign ^ io.subOp)
  val eqSigns0      = RegNext(io.a.sign === (io.b.sign ^ io.subOp))
  val sDiffExps0    = RegNext(io.a.sExp - io.b.sExp)
  val roundingMode0 = RegNext(io.roundingMode)

  // Stage 1: prepare close-path subtraction and far-path operand selection.
  val modNatAlignDist1 = Mux(sDiffExps0 < 0.S, -sDiffExps0, sDiffExps0)(alignDistWidth - 1, 0)

  val isMaxAlign1 =
    (sDiffExps0 >> alignDistWidth) =/= 0.S &&
      ((sDiffExps0 >> alignDistWidth) =/= -1.S || sDiffExps0(alignDistWidth - 1, 0) === 0.U)

  val alignDist1    = Mux(isMaxAlign1, ((BigInt(1) << alignDistWidth) - 1).U, modNatAlignDist1)
  val closeSubMags1 = !eqSigns0 && !isMaxAlign1 && (modNatAlignDist1 <= 1.U)

  val closeAlignedSigA1 =
    Mux((0.S <= sDiffExps0) && sDiffExps0(0), a0.sig << 2, 0.U) |
      Mux((0.S <= sDiffExps0) && !sDiffExps0(0), a0.sig << 1, 0.U) |
      Mux(sDiffExps0 < 0.S, a0.sig, 0.U)

  val closeSSigSum1 = Wire(SInt((sigWidth + 3).W))
  closeSSigSum1 := closeAlignedSigA1.asSInt - (b0.sig << 1).asSInt
  val closeSigSum1 = Wire(UInt((sigWidth + 2).W))
  closeSigSum1 := Mux(closeSSigSum1 < 0.S, -closeSSigSum1, closeSSigSum1)(sigWidth + 1, 0)
  val closeAdjustedSigSum1 = Wire(UInt((sigWidth + 2).W))
  closeAdjustedSigSum1 := closeSigSum1 << (sigWidth & 1)
  val closeReduced2SigSum1 = orReduceBy2(closeAdjustedSigSum1)

  val farSignOut1   = Mux(sDiffExps0 < 0.S, effSignB0, a0.sign)
  val farSigLarger1 = Wire(UInt(sigWidth.W))
  farSigLarger1 := Mux(sDiffExps0 < 0.S, b0.sig, a0.sig)(sigWidth - 1, 0)
  val farSigSmaller1 = Wire(UInt(sigWidth.W))
  farSigSmaller1 := Mux(sDiffExps0 < 0.S, a0.sig, b0.sig)(sigWidth - 1, 0)

  val a1            = RegNext(a0)
  val b1            = RegNext(b0)
  val effSignB1     = RegNext(effSignB0)
  val eqSigns1      = RegNext(eqSigns0)
  val sDiffExps1    = RegNext(sDiffExps0)
  val roundingMode1 = RegNext(roundingMode0)
  val alignDist1r   = RegInit(0.U(alignDistWidth.W))
  alignDist1r := alignDist1
  val closeSubMags1r = RegInit(false.B)
  closeSubMags1r := closeSubMags1
  val closeSSigSum1r = RegInit(0.S((sigWidth + 3).W))
  closeSSigSum1r := closeSSigSum1
  val closeSigSum1r = RegInit(0.U((sigWidth + 2).W))
  closeSigSum1r := closeSigSum1
  val closeReduced2SigSum1r = RegInit(0.U(((sigWidth + 3) / 2).W))
  closeReduced2SigSum1r := closeReduced2SigSum1
  val farSignOut1r   = RegNext(farSignOut1)
  val farSigLarger1r = RegInit(0.U(sigWidth.W))
  farSigLarger1r := farSigLarger1
  val farSigSmaller1r = RegInit(0.U(sigWidth.W))
  farSigSmaller1r := farSigSmaller1

  // Stage 2: leading-zero normalization and far-path alignment.
  val closeNormDistReduced2_2           = countLeadingZeros(closeReduced2SigSum1r)
  val closeNearNormDist2                = (closeNormDistReduced2_2 << 1)(alignDistWidth - 1, 0)
  val closeSigOut2                      = ((closeSigSum1r << closeNearNormDist2) << 1)(sigWidth + 2, 0)
  val closeTotalCancellation2           = !(closeSigOut2(sigWidth + 2, sigWidth + 1).orR)
  val closeNotTotalCancellationSignOut2 = a1.sign ^ (closeSSigSum1r < 0.S)

  val farMainAlignedSigSmaller2 = Wire(UInt((sigWidth + 5).W))
  farMainAlignedSigSmaller2 := (farSigSmaller1r << 5) >> alignDist1r
  val farReduced4SigSmaller2 = Wire(UInt(((sigWidth + 5) / 4).W))
  farReduced4SigSmaller2 := orReduceBy4(Cat(farSigSmaller1r, 0.U(2.W)))
  val farRoundExtraMask2 = lowMask(alignDist1r(alignDistWidth - 1, 2), (sigWidth + 5) / 4, 0)

  val farAlignedSigSmaller2 = Cat(
    farMainAlignedSigSmaller2 >> 3,
    farMainAlignedSigSmaller2(2, 0).orR || (farReduced4SigSmaller2 & farRoundExtraMask2).orR
  )

  val farSubMags2 = !eqSigns1

  val a2                                 = RegNext(a1)
  val b2                                 = RegNext(b1)
  val effSignB2                          = RegNext(effSignB1)
  val eqSigns2                           = RegNext(eqSigns1)
  val sDiffExps2                         = RegNext(sDiffExps1)
  val roundingMode2                      = RegNext(roundingMode1)
  val closeSubMags2r                     = RegNext(closeSubMags1r)
  val closeNearNormDist2r                = RegNext(closeNearNormDist2)
  val closeSigOut2r                      = RegNext(closeSigOut2)
  val closeTotalCancellation2r           = RegNext(closeTotalCancellation2)
  val closeNotTotalCancellationSignOut2r = RegNext(closeNotTotalCancellationSignOut2)
  val farSignOut2r                       = RegNext(farSignOut1r)
  val farSigLarger2r                     = RegNext(farSigLarger1r)
  val farAlignedSigSmaller2r             = RegNext(farAlignedSigSmaller2)
  val farSubMags2r                       = RegNext(farSubMags2)

  // Stage 3: significand add/subtract and special-case selection.
  val farNegAlignedSigSmaller3 = Mux(
    farSubMags2r,
    Cat(1.U, ~farAlignedSigSmaller2r),
    farAlignedSigSmaller2r
  )

  val farSigSum3 = (farSigLarger2r << 3) + farNegAlignedSigSmaller3 + farSubMags2r
  val farSigOut3 = Mux(farSubMags2r, farSigSum3, (farSigSum3 >> 1) | farSigSum3(0))(sigWidth + 2, 0)

  val notSigNaNInvalidExc3 = a2.isInf && b2.isInf && !eqSigns2
  val notNaNIsInfOut3      = a2.isInf || b2.isInf
  val addZeros3            = a2.isZero && b2.isZero
  val notNaNSpecialCase3   = notNaNIsInfOut3 || addZeros3
  val notNaNIsZeroOut3     = addZeros3 || (!notNaNIsInfOut3 && closeSubMags2r && closeTotalCancellation2r)

  val notNaNSignOut3 =
    (eqSigns2 && a2.sign) ||
      (a2.isInf && a2.sign) ||
      (b2.isInf && effSignB2) ||
      (notNaNIsZeroOut3 && !eqSigns2 && roundingMode2 === round_min) ||
      (!notNaNSpecialCase3 && closeSubMags2r && !closeTotalCancellation2r && closeNotTotalCancellationSignOut2r) ||
      (!notNaNSpecialCase3 && !closeSubMags2r && farSignOut2r)

  val commonSExp3 =
    Mux(closeSubMags2r || (sDiffExps2 < 0.S), b2.sExp, a2.sExp) -
      Mux(closeSubMags2r, closeNearNormDist2r, farSubMags2r).zext

  val commonSigOut3 = Mux(closeSubMags2r, closeSigOut2r, farSigOut3)

  val rawOutReg = RegInit(0.U.asTypeOf(new RawFloat(expWidth, sigWidth + 2)))
  rawOutReg.isNaN  := a2.isNaN || b2.isNaN
  rawOutReg.isInf  := notNaNIsInfOut3
  rawOutReg.isZero := notNaNIsZeroOut3
  rawOutReg.sExp   := commonSExp3
  rawOutReg.sign   := notNaNSignOut3
  rawOutReg.sig    := commonSigOut3

  io.rawOut     := rawOutReg
  io.invalidExc := RegNext(isSigNaNRawFloat(a2) || isSigNaNRawFloat(b2) || notSigNaNInvalidExc3)
  io.validout   := ShiftRegister(io.validin, 4)
}
