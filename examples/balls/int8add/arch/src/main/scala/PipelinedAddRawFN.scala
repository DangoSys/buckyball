package examples.balls.int8add

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
class PipelinedAddRawFN(expWidth: Int, sigWidth: Int) extends Module {

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

  private val alignDistWidth = log2Ceil(sigWidth)

  // Stage 0: register the raw operands and exponent comparison.
  private val a0            = RegNext(io.a)
  private val b0            = RegNext(io.b)
  private val effSignB0     = RegNext(io.b.sign ^ io.subOp)
  private val eqSigns0      = RegNext(io.a.sign === (io.b.sign ^ io.subOp))
  private val sDiffExps0    = RegNext(io.a.sExp - io.b.sExp)
  private val roundingMode0 = RegNext(io.roundingMode)

  // Stage 1: prepare close-path subtraction and far-path operand selection.
  private val modNatAlignDist1 = Mux(sDiffExps0 < 0.S, -sDiffExps0, sDiffExps0)(alignDistWidth - 1, 0)

  private val isMaxAlign1 =
    (sDiffExps0 >> alignDistWidth) =/= 0.S &&
      ((sDiffExps0 >> alignDistWidth) =/= -1.S || sDiffExps0(alignDistWidth - 1, 0) === 0.U)

  private val alignDist1    = Mux(isMaxAlign1, ((BigInt(1) << alignDistWidth) - 1).U, modNatAlignDist1)
  private val closeSubMags1 = !eqSigns0 && !isMaxAlign1 && (modNatAlignDist1 <= 1.U)

  private val closeAlignedSigA1 =
    Mux((0.S <= sDiffExps0) && sDiffExps0(0), a0.sig << 2, 0.U) |
      Mux((0.S <= sDiffExps0) && !sDiffExps0(0), a0.sig << 1, 0.U) |
      Mux(sDiffExps0 < 0.S, a0.sig, 0.U)

  private val closeSSigSum1 = Wire(SInt((sigWidth + 3).W))
  closeSSigSum1 := closeAlignedSigA1.asSInt - (b0.sig << 1).asSInt
  private val closeSigSum1 = Wire(UInt((sigWidth + 2).W))
  closeSigSum1 := Mux(closeSSigSum1 < 0.S, -closeSSigSum1, closeSSigSum1)(sigWidth + 1, 0)
  private val closeAdjustedSigSum1 = Wire(UInt((sigWidth + 2).W))
  closeAdjustedSigSum1 := closeSigSum1 << (sigWidth & 1)
  private val closeReduced2SigSum1 = orReduceBy2(closeAdjustedSigSum1)

  private val farSignOut1   = Mux(sDiffExps0 < 0.S, effSignB0, a0.sign)
  private val farSigLarger1 = Wire(UInt(sigWidth.W))
  farSigLarger1 := Mux(sDiffExps0 < 0.S, b0.sig, a0.sig)(sigWidth - 1, 0)
  private val farSigSmaller1 = Wire(UInt(sigWidth.W))
  farSigSmaller1 := Mux(sDiffExps0 < 0.S, a0.sig, b0.sig)(sigWidth - 1, 0)

  private val a1            = RegNext(a0)
  private val b1            = RegNext(b0)
  private val effSignB1     = RegNext(effSignB0)
  private val eqSigns1      = RegNext(eqSigns0)
  private val sDiffExps1    = RegNext(sDiffExps0)
  private val roundingMode1 = RegNext(roundingMode0)
  private val alignDist1r   = RegInit(0.U(alignDistWidth.W))
  alignDist1r := alignDist1
  private val closeSubMags1r = RegInit(false.B)
  closeSubMags1r := closeSubMags1
  private val closeSSigSum1r = RegInit(0.S((sigWidth + 3).W))
  closeSSigSum1r := closeSSigSum1
  private val closeSigSum1r = RegInit(0.U((sigWidth + 2).W))
  closeSigSum1r := closeSigSum1
  private val closeReduced2SigSum1r = RegInit(0.U(((sigWidth + 3) / 2).W))
  closeReduced2SigSum1r := closeReduced2SigSum1
  private val farSignOut1r   = RegNext(farSignOut1)
  private val farSigLarger1r = RegInit(0.U(sigWidth.W))
  farSigLarger1r := farSigLarger1
  private val farSigSmaller1r = RegInit(0.U(sigWidth.W))
  farSigSmaller1r := farSigSmaller1

  // Stage 2: leading-zero normalization and far-path alignment.
  private val closeNormDistReduced2_2           = countLeadingZeros(closeReduced2SigSum1r)
  private val closeNearNormDist2                = (closeNormDistReduced2_2 << 1)(alignDistWidth - 1, 0)
  private val closeSigOut2                      = ((closeSigSum1r << closeNearNormDist2) << 1)(sigWidth + 2, 0)
  private val closeTotalCancellation2           = !(closeSigOut2(sigWidth + 2, sigWidth + 1).orR)
  private val closeNotTotalCancellationSignOut2 = a1.sign ^ (closeSSigSum1r < 0.S)

  private val farMainAlignedSigSmaller2 = Wire(UInt((sigWidth + 5).W))
  farMainAlignedSigSmaller2 := (farSigSmaller1r << 5) >> alignDist1r
  private val farReduced4SigSmaller2 = Wire(UInt(((sigWidth + 5) / 4).W))
  farReduced4SigSmaller2 := orReduceBy4(Cat(farSigSmaller1r, 0.U(2.W)))
  private val farRoundExtraMask2 = lowMask(alignDist1r(alignDistWidth - 1, 2), (sigWidth + 5) / 4, 0)

  private val farAlignedSigSmaller2 = Cat(
    farMainAlignedSigSmaller2 >> 3,
    farMainAlignedSigSmaller2(2, 0).orR || (farReduced4SigSmaller2 & farRoundExtraMask2).orR
  )

  private val farSubMags2 = !eqSigns1

  private val a2                                 = RegNext(a1)
  private val b2                                 = RegNext(b1)
  private val effSignB2                          = RegNext(effSignB1)
  private val eqSigns2                           = RegNext(eqSigns1)
  private val sDiffExps2                         = RegNext(sDiffExps1)
  private val roundingMode2                      = RegNext(roundingMode1)
  private val closeSubMags2r                     = RegNext(closeSubMags1r)
  private val closeNearNormDist2r                = RegNext(closeNearNormDist2)
  private val closeSigOut2r                      = RegNext(closeSigOut2)
  private val closeTotalCancellation2r           = RegNext(closeTotalCancellation2)
  private val closeNotTotalCancellationSignOut2r = RegNext(closeNotTotalCancellationSignOut2)
  private val farSignOut2r                       = RegNext(farSignOut1r)
  private val farSigLarger2r                     = RegNext(farSigLarger1r)
  private val farAlignedSigSmaller2r             = RegNext(farAlignedSigSmaller2)
  private val farSubMags2r                       = RegNext(farSubMags2)

  // Stage 3: significand add/subtract and special-case selection.
  private val farNegAlignedSigSmaller3 = Mux(
    farSubMags2r,
    Cat(1.U, ~farAlignedSigSmaller2r),
    farAlignedSigSmaller2r
  )

  private val farSigSum3 = (farSigLarger2r << 3) + farNegAlignedSigSmaller3 + farSubMags2r
  private val farSigOut3 = Mux(farSubMags2r, farSigSum3, (farSigSum3 >> 1) | farSigSum3(0))(sigWidth + 2, 0)

  private val notSigNaNInvalidExc3 = a2.isInf && b2.isInf && !eqSigns2
  private val notNaNIsInfOut3      = a2.isInf || b2.isInf
  private val addZeros3            = a2.isZero && b2.isZero
  private val notNaNSpecialCase3   = notNaNIsInfOut3 || addZeros3
  private val notNaNIsZeroOut3     = addZeros3 || (!notNaNIsInfOut3 && closeSubMags2r && closeTotalCancellation2r)

  private val notNaNSignOut3 =
    (eqSigns2 && a2.sign) ||
      (a2.isInf && a2.sign) ||
      (b2.isInf && effSignB2) ||
      (notNaNIsZeroOut3 && !eqSigns2 && roundingMode2 === round_min) ||
      (!notNaNSpecialCase3 && closeSubMags2r && !closeTotalCancellation2r && closeNotTotalCancellationSignOut2r) ||
      (!notNaNSpecialCase3 && !closeSubMags2r && farSignOut2r)

  private val commonSExp3 =
    Mux(closeSubMags2r || (sDiffExps2 < 0.S), b2.sExp, a2.sExp) -
      Mux(closeSubMags2r, closeNearNormDist2r, farSubMags2r).zext

  private val commonSigOut3 = Mux(closeSubMags2r, closeSigOut2r, farSigOut3)

  private val rawOutReg = RegInit(0.U.asTypeOf(new RawFloat(expWidth, sigWidth + 2)))
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
