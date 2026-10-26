// See LICENSE.Berkeley and LICENSE.SiFive for license details.
package framework.lib.arith

import chisel3.experimental.hierarchy.{Instance, Instantiate}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

@instantiable
class IntToFloat(intWidth: Int, expWidth: Int, sigWidth: Int) extends Module {

  @public
  val io = IO(new Bundle {
    val validin        = Input(Bool())
    val signedIn       = Input(Bool())
    val in             = Input(Bits(intWidth.W))
    val roundingMode   = Input(UInt(3.W))
    val detectTininess = Input(UInt(1.W))
    val out            = Output(Bits((expWidth + sigWidth + 1).W))
    val exceptionFlags = Output(Bits(5.W))
    val validout       = Output(Bool())
  })

  val rawExpWidth         = log2Up(intWidth) + 1
  val extIntWidth         = 1 << (rawExpWidth - 1)
  val normalizationLevels = rawExpWidth - 1
  val firstStageLevels    = 2
  val secondStageLevels   = normalizationLevels - firstStageLevels
  require(secondStageLevels > 0)

  val sign     = io.signedIn && io.in(intWidth - 1)
  val absIn    = Mux(sign, -io.in.asUInt, io.in.asUInt)
  val extAbsIn = (0.U(extIntWidth.W) ## absIn)(extIntWidth - 1, 0)

  // Complete the 32- and 16-bit normalization decisions before the cut.
  val firstNormStages = Wire(Vec(firstStageLevels + 1, UInt(extIntWidth.W)))
  val firstNormDist   = Wire(Vec(firstStageLevels, Bool()))
  firstNormStages(0) := extAbsIn
  for (level <- 0 until firstStageLevels) {
    val shiftAmt        = extIntWidth >> (level + 1)
    val upperHalfIsZero =
      !firstNormStages(level)(extIntWidth - 1, extIntWidth - shiftAmt).orR
    firstNormStages(level + 1)                  := Mux(
      upperHalfIsZero,
      (firstNormStages(level) << shiftAmt)(extIntWidth - 1, 0),
      firstNormStages(level)
    )
    firstNormDist(firstStageLevels - 1 - level) := upperHalfIsZero
  }

  val midNormStage      = RegEnable(firstNormStages(firstStageLevels), io.validin)
  val midNormDist       = RegEnable(firstNormDist.asUInt, io.validin)
  val midSign           = RegEnable(sign, io.validin)
  val midRoundingMode   = RegEnable(io.roundingMode, io.validin)
  val midDetectTininess = RegEnable(io.detectTininess, io.validin)
  val midValid          = RegNext(io.validin, false.B)

  val secondNormStages = Wire(Vec(secondStageLevels + 1, UInt(extIntWidth.W)))
  val secondNormDist   = Wire(Vec(secondStageLevels, Bool()))
  secondNormStages(0) := midNormStage
  for (offset <- 0 until secondStageLevels) {
    val level           = firstStageLevels + offset
    val shiftAmt        = extIntWidth >> (level + 1)
    val upperHalfIsZero =
      !secondNormStages(offset)(extIntWidth - 1, extIntWidth - shiftAmt).orR
    secondNormStages(offset + 1)                   := Mux(
      upperHalfIsZero,
      (secondNormStages(offset) << shiftAmt)(extIntWidth - 1, 0),
      secondNormStages(offset)
    )
    secondNormDist(secondStageLevels - 1 - offset) := upperHalfIsZero
  }

  val adjustedNormDist = Cat(midNormDist, secondNormDist.asUInt)
  val sig              = secondNormStages(secondStageLevels)(extIntWidth - 1, extIntWidth - intWidth)
  val intAsRawFloat    = Wire(new hardfloat.RawFloat(rawExpWidth, intWidth))
  intAsRawFloat.isNaN  := false.B
  intAsRawFloat.isInf  := false.B
  intAsRawFloat.isZero := !sig(intWidth - 1)
  intAsRawFloat.sign   := midSign
  intAsRawFloat.sExp   := (2.U(2.W) ## ~adjustedNormDist).zext
  intAsRawFloat.sig    := sig

  // Isolate the remaining normalization and raw-float construction from the
  // target-format rounding and packing logic.
  val roundInput          = RegEnable(intAsRawFloat, midValid)
  val roundRoundingMode   = RegEnable(midRoundingMode, midValid)
  val roundDetectTininess = RegEnable(midDetectTininess, midValid)
  val roundValid          = RegNext(midValid, false.B)

  val roundAnyRawFNToRecFN = Instantiate(new hardfloat.RoundAnyRawFNToRecFN(
    rawExpWidth,
    intWidth,
    expWidth,
    sigWidth,
    hardfloat.consts.flRoundOpt_sigMSBitAlwaysZero | hardfloat.consts.flRoundOpt_neverUnderflows
  ))

  roundAnyRawFNToRecFN.io.invalidExc     := false.B
  roundAnyRawFNToRecFN.io.infiniteExc    := false.B
  roundAnyRawFNToRecFN.io.in             := roundInput
  roundAnyRawFNToRecFN.io.roundingMode   := roundRoundingMode
  roundAnyRawFNToRecFN.io.detectTininess := roundDetectTininess

  io.out            := roundAnyRawFNToRecFN.io.out
  io.exceptionFlags := roundAnyRawFNToRecFN.io.exceptionFlags
  io.validout       := roundValid
}
