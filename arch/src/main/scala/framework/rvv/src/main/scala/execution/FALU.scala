package framework.rvv

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import hardfloat._

@instantiable
class FALU extends Module {

  @public val io = IO(new Bundle {
    val a            = Input(UInt(64.W))
    val b            = Input(UInt(64.W))
    val c            = Input(UInt(64.W))
    val sew          = Input(UInt(2.W))
    val op           = Input(UInt(6.W))
    val subop        = Input(UInt(5.W))
    val start        = Input(Bool())
    val roundingMode = Input(UInt(3.W))
    val ready        = Output(Bool())
    val valid        = Output(Bool())
    val result       = Output(UInt(64.W))
    val flags        = Output(UInt(5.W))
    val legal        = Output(Bool())
  })

  val squareRoot    = io.op === 19.U && io.subop === 0.U
  val longOperation = io.op === 32.U || io.op === 33.U || squareRoot
  val results       = Wire(Vec(2, UInt(64.W)))
  val flags         = Wire(Vec(2, UInt(5.W)))
  val ready         = Wire(Vec(2, Bool()))
  val valid         = Wire(Vec(2, Bool()))
  for ((expWidth, sigWidth, index) <- Seq((8, 24, 0), (11, 53, 1))) {
    val width             = expWidth + sigWidth
    val fractionWidth     = sigWidth - 1
    val bias              = (1 << (expWidth - 1)) - 1
    val canonicalNaN      = ((BigInt(2 * bias + 1) << fractionWidth) | (BigInt(1) << (fractionWidth - 1))).U(width.W)
    val infinityMagnitude = (BigInt(2 * bias + 1) << fractionWidth).U((width - 1).W)
    val maximumMagnitude  = ((BigInt(2 * bias + 1) << fractionWidth) - 1).U((width - 1).W)
    val av                = io.a(width - 1, 0)
    val bv                = io.b(width - 1, 0)
    val cv                = io.c(width - 1, 0)
    val a                 = recFNFromFN(expWidth, sigWidth, av)
    val b                 = recFNFromFN(expWidth, sigWidth, bv)
    val c                 = recFNFromFN(expWidth, sigWidth, cv)
    val add               = Module(new AddRecFN(expWidth, sigWidth))
    add.io.a              := Mux(io.op === 39.U, b, a)
    add.io.b              := Mux(io.op === 39.U, a, b)
    add.io.subOp          := io.op === 2.U || io.op === 39.U
    add.io.roundingMode   := io.roundingMode
    add.io.detectTininess := 1.U
    val mul = Module(new MulRecFN(expWidth, sigWidth))
    mul.io.a              := a
    mul.io.b              := b
    mul.io.roundingMode   := io.roundingMode
    mul.io.detectTininess := 1.U
    val fma = Module(new MulAddRecFN(expWidth, sigWidth))
    fma.io.a              := b
    fma.io.b              := Mux(io.op(2), a, c)
    fma.io.c              := Mux(io.op(2), c, a)
    fma.io.op             := Cat(io.op(0), io.op(1) ^ io.op(0))
    fma.io.roundingMode   := io.roundingMode
    fma.io.detectTininess := 1.U
    val cmp = Module(new CompareRecFN(expWidth, sigWidth))
    cmp.io.a         := a
    cmp.io.b         := b
    cmp.io.signaling := Seq(25, 27, 29, 31).map(n => io.op === n.U).reduce(_ || _)
    val div = Module(new DivSqrtRecFN_small(expWidth, sigWidth, 0))
    div.io.a              := Mux(io.op === 33.U, b, a)
    div.io.b              := Mux(io.op === 33.U, a, b)
    div.io.sqrtOp         := squareRoot
    div.io.roundingMode   := io.roundingMode
    div.io.detectTininess := 1.U
    div.io.inValid        := io.start && io.legal && longOperation && io.sew === (index + 2).U && div.io.inReady
    val aNaN            = av(width - 2, fractionWidth).andR && av(fractionWidth - 1, 0).orR
    val bNaN            = bv(width - 2, fractionWidth).andR && bv(fractionWidth - 1, 0).orR
    val aSNaN           = aNaN && !av(fractionWidth - 1)
    val bSNaN           = bNaN && !bv(fractionWidth - 1)
    val aZero           = !av(width - 2, 0).orR
    val aInf            = av(width - 2, fractionWidth).andR && !av(fractionWidth - 1, 0).orR
    val bothZero        = aZero && !bv(width - 2, 0).orR
    val minimum         =
      Mux(aNaN && bNaN, canonicalNaN, Mux(aNaN, bv, Mux(bNaN, av, Mux(bothZero, av | bv, Mux(cmp.io.lt, av, bv)))))
    val maximum         =
      Mux(aNaN && bNaN, canonicalNaN, Mux(aNaN, bv, Mux(bNaN, av, Mux(bothZero, av & bv, Mux(cmp.io.gt, av, bv)))))
    // RVV 1.0 vfrec7/vfrsqrt7 tables; indices use the normalized input fraction.
    val reciprocalTable = VecInit(Seq(
      127, 125, 123, 121, 119, 117, 116, 114, 112, 110, 109, 107, 105, 104, 102, 100, 99, 97, 96, 94, 93, 91, 90, 88,
      87, 85, 84, 83, 81, 80, 79, 77, 76, 75, 74, 72, 71, 70, 69, 68, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55,
      54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 40, 39, 38, 37, 36, 35, 35, 34, 33, 32, 31, 31, 30,
      29, 28, 28, 27, 26, 25, 25, 24, 23, 23, 22, 21, 21, 20, 19, 19, 18, 17, 17, 16, 15, 15, 14, 14, 13, 12, 12, 11,
      11, 10, 9, 9, 8, 8, 7, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0
    ).map(_.U(7.W)))

    val reciprocalSqrtTable = VecInit(Seq(
      52, 51, 50, 48, 47, 46, 44, 43, 42, 41, 40, 39, 38, 36, 35, 34, 33, 32, 31, 30, 30, 29, 28, 27, 26, 25, 24, 23,
      23, 22, 21, 20, 19, 19, 18, 17, 16, 16, 15, 14, 14, 13, 12, 12, 11, 10, 10, 9, 9, 8, 7, 7, 6, 6, 5, 4, 4, 3, 3, 2,
      2, 1, 1, 0, 127, 125, 123, 121, 119, 118, 116, 114, 113, 111, 109, 108, 106, 105, 103, 102, 100, 99, 97, 96, 95,
      93, 92, 91, 90, 88, 87, 86, 85, 84, 83, 82, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 70, 69, 68, 67, 66, 65,
      64, 63, 63, 62, 61, 60, 59, 59, 58, 57, 56, 56, 55, 54, 53
    ).map(_.U(7.W)))

    val leadingZeros       = PriorityEncoder(Reverse(av(fractionWidth - 1, 0)))
    val normalizedExponent =
      Mux(av(width - 2, fractionWidth).orR, Cat(0.U(1.W), av(width - 2, fractionWidth)).asSInt, -leadingZeros.zext)
    val normalizedFraction = Mux(
      av(width - 2, fractionWidth).orR,
      av(fractionWidth - 1, 0),
      (av(fractionWidth - 1, 0) << (leadingZeros +& 1.U))(fractionWidth - 1, 0)
    )
    val reciprocalExponent = (2 * bias - 1).S((expWidth + 2).W) - normalizedExponent
    val reciprocalFraction =
      Cat(reciprocalTable(normalizedFraction(fractionWidth - 1, fractionWidth - 7)), 0.U((fractionWidth - 7).W))

    val reciprocalFinite = Cat(
      av(width - 1),
      Mux(reciprocalExponent <= 0.S, 0.U(expWidth.W), reciprocalExponent.asUInt(expWidth - 1, 0)),
      Mux(
        reciprocalExponent <= 0.S,
        (Cat(1.U(1.W), reciprocalFraction) >> Mux(reciprocalExponent === 0.S, 1.U, 2.U))(fractionWidth - 1, 0),
        reciprocalFraction
      )
    )

    val reciprocalOverflow = reciprocalExponent > (2 * bias).S
    val overflowToFinite   = io.roundingMode === 1.U ||
      (io.roundingMode === 2.U && !av(width - 1)) || (io.roundingMode === 3.U && av(width - 1))
    val signedInfinity     = Cat(av(width - 1), infinityMagnitude)

    val reciprocal = Mux(
      aNaN,
      canonicalNaN,
      Mux(
        aInf,
        Cat(av(width - 1), 0.U((width - 1).W)),
        Mux(
          aZero,
          signedInfinity,
          Mux(
            reciprocalOverflow,
            Mux(overflowToFinite, Cat(av(width - 1), maximumMagnitude), signedInfinity),
            reciprocalFinite
          )
        )
      )
    )

    val reciprocalFlags =
      Mux(aNaN, Cat(aSNaN, 0.U(4.W)), Mux(aInf, 0.U, Mux(aZero, 8.U, Mux(reciprocalOverflow, 5.U, 0.U))))
    val sqrtExponent    = (((3 * bias - 1).S((expWidth + 2).W) - normalizedExponent) >> 1).asUInt
    val sqrtFraction    =
      reciprocalSqrtTable(Cat(normalizedExponent.asUInt(0), normalizedFraction(fractionWidth - 1, fractionWidth - 6)))
    val sqrtInvalid     = aSNaN || (!aNaN && av(width - 1) && !aZero)

    val reciprocalSqrt = Mux(
      aNaN || sqrtInvalid,
      canonicalNaN,
      Mux(
        aZero,
        signedInfinity,
        Mux(aInf, 0.U, Cat(av(width - 1), sqrtExponent(expWidth - 1, 0), sqrtFraction, 0.U((fractionWidth - 7).W)))
      )
    )

    val reciprocalSqrtFlags = Mux(sqrtInvalid, 16.U, Mux(aZero, 8.U, 0.U))

    results(index) := 0.U
    flags(index)   := 0.U
    ready(index)   := div.io.inReady
    valid(index)   := Mux(
      longOperation,
      div.io.outValid_div || div.io.outValid_sqrt,
      io.start && io.legal && div.io.inReady
    )
    switch(io.op) {
      is(0.U, 1.U, 2.U, 3.U, 39.U) {
        results(index) := fNFromRecFN(expWidth, sigWidth, add.io.out); flags(index) := add.io.exceptionFlags
      }
      is(4.U, 5.U) { results(index) := minimum; flags(index) := Cat(aSNaN || bSNaN, 0.U(4.W)) }
      is(6.U, 7.U) { results(index) := maximum; flags(index) := Cat(aSNaN || bSNaN, 0.U(4.W)) }
      is(8.U)(results(index)  := Cat(bv(width - 1), av(width - 2, 0)))
      is(9.U)(results(index)  := Cat(!bv(width - 1), av(width - 2, 0)))
      is(10.U)(results(index) := Cat(av(width - 1) ^ bv(width - 1), av(width - 2, 0)))
      is(19.U) {
        switch(io.subop) {
          is(0.U) {
            results(index) := fNFromRecFN(expWidth, sigWidth, div.io.out); flags(index) := div.io.exceptionFlags
          }
          is(4.U) { results(index) := reciprocalSqrt; flags(index) := reciprocalSqrtFlags }
          is(5.U) { results(index) := reciprocal; flags(index) := reciprocalFlags }
          is(16.U)(results(index) := classifyRecFN(expWidth, sigWidth, a))
        }
      }
      is(24.U) { results(index) := cmp.io.eq; flags(index) := cmp.io.exceptionFlags }
      is(25.U) { results(index) := cmp.io.lt || cmp.io.eq; flags(index) := cmp.io.exceptionFlags }
      is(27.U) { results(index) := cmp.io.lt; flags(index) := cmp.io.exceptionFlags }
      is(28.U) { results(index) := !cmp.io.eq; flags(index) := cmp.io.exceptionFlags }
      is(29.U) { results(index) := cmp.io.gt; flags(index) := cmp.io.exceptionFlags }
      is(31.U) { results(index) := cmp.io.gt || cmp.io.eq; flags(index) := cmp.io.exceptionFlags }
      is(32.U, 33.U) {
        results(index) := fNFromRecFN(expWidth, sigWidth, div.io.out); flags(index) := div.io.exceptionFlags
      }
      is(36.U) { results(index) := fNFromRecFN(expWidth, sigWidth, mul.io.out); flags(index) := mul.io.exceptionFlags }
      is(40.U, 41.U, 42.U, 43.U, 44.U, 45.U, 46.U, 47.U) {
        results(index) := fNFromRecFN(expWidth, sigWidth, fma.io.out); flags(index) := fma.io.exceptionFlags
      }
    }
  }
  val selectedResult = WireDefault(results(io.sew(0)))
  io.flags := flags(io.sew(0))
  io.ready := ready(io.sew(0))
  io.valid := valid(io.sew(0))
  val conversionLegal = WireDefault(false.B)
  when(io.op === 18.U) {
    selectedResult := 0.U
    io.flags       := 0.U
  }
  for ((srcWidth, dstWidth, baseSelector) <- Seq((32, 32, 0), (64, 64, 0), (32, 64, 8), (64, 32, 16))) {
    val (se, ss)     = if (srcWidth == 32) (8, 24) else (11, 53)
    val (de, ds)     = if (dstWidth == 32) (8, 24) else (11, 53)
    val matchesWidth = io.sew === (if (baseSelector != 0 || srcWidth == 32) 2 else 3).U
    val floatToInt   = Module(new RecFNToIN(se, ss, dstWidth))
    floatToInt.io.in           := recFNFromFN(se, ss, io.a(srcWidth - 1, 0))
    floatToInt.io.signedOut    := io.subop(0)
    floatToInt.io.roundingMode := Mux(io.subop(2, 0) >= 6.U, 1.U, io.roundingMode)
    val intToFloat = Module(new INToRecFN(srcWidth, de, ds))
    intToFloat.io.in             := io.a(srcWidth - 1, 0)
    intToFloat.io.signedIn       := io.subop(0)
    intToFloat.io.roundingMode   := io.roundingMode
    intToFloat.io.detectTininess := 1.U
    val integerSelectors  = Seq(0, 1, 6, 7).map(n => io.subop === (baseSelector + n).U).reduce(_ || _)
    val floatingSelectors = Seq(2, 3).map(n => io.subop === (baseSelector + n).U).reduce(_ || _)
    when(matchesWidth && (integerSelectors || floatingSelectors))(conversionLegal := true.B)
    when(io.op === 18.U && matchesWidth && integerSelectors) {
      selectedResult := floatToInt.io.out
      io.flags       := Cat(floatToInt.io.intExceptionFlags(2, 1).orR, 0.U(3.W), floatToInt.io.intExceptionFlags(0))
    }
    when(io.op === 18.U && matchesWidth && floatingSelectors) {
      selectedResult := fNFromRecFN(de, ds, intToFloat.io.out)
      io.flags       := intToFloat.io.exceptionFlags
    }
    if (baseSelector != 0) {
      val convert = Module(new RecFNToRecFN(se, ss, de, ds))
      convert.io.in             := recFNFromFN(se, ss, io.a(srcWidth - 1, 0))
      convert.io.roundingMode   := Mux(io.subop === 21.U, 6.U, io.roundingMode)
      convert.io.detectTininess := 1.U
      val matches = matchesWidth && (io.subop === (baseSelector + 4).U || (baseSelector == 16).B && io.subop === 21.U)
      when(matches)(conversionLegal := true.B)
      when(io.op === 18.U && matches) {
        selectedResult := fNFromRecFN(de, ds, convert.io.out)
        io.flags       := convert.io.exceptionFlags
      }
    }
  }

  val floatingResult = !Seq(8, 9, 10, 24, 25, 27, 28, 29, 31).map(n => io.op === n.U).reduce(_ || _) &&
    !(io.op === 19.U && io.subop === 16.U) &&
    !(io.op === 18.U && (io.subop(2, 0) === 0.U || io.subop(2, 0) === 1.U || io.subop(2, 0) >= 6.U))

  val destinationDouble =
    Mux(io.op === 18.U, Mux(io.subop >= 16.U, false.B, Mux(io.subop >= 8.U, true.B, io.sew === 3.U)), io.sew === 3.U)
  val rawResult         = selectedResult

  val finalResult = Mux(
    destinationDouble,
    Mux(rawResult(62, 52).andR && rawResult(51, 0).orR, "h7ff8000000000000".U, rawResult),
    Mux(rawResult(30, 23).andR && rawResult(22, 0).orR, "h7fc00000".U, rawResult)
  )

  io.result := Mux(floatingResult, finalResult, selectedResult)
  io.legal  := io.sew >= 2.U && io.roundingMode <= 4.U && (
    Seq(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 24, 25, 27, 28, 29, 31, 32, 33, 36, 39, 40, 41, 42, 43, 44, 45, 46, 47).map(
      n => io.op === n.U
    ).reduce(_ || _) ||
      (io.op === 18.U && conversionLegal) || (io.op === 19.U && Seq(0, 4, 5, 16).map(n => io.subop === n.U).reduce(
        _ || _
      ))
  )
}
