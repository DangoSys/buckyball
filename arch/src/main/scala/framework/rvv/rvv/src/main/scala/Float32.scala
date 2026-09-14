package framework.rvv

import chisel3._
import chisel3.util._
import hardfloat._

/** Shared Zve32f datapath: a=vs2, b=vs1/scalar, c=old vd; op=funct6, subop=rs1. */
class Float32 extends Module {

  val io = IO(new Bundle {
    val a            = Input(UInt(32.W))
    val b            = Input(UInt(32.W))
    val c            = Input(UInt(32.W))
    val op           = Input(UInt(6.W))
    val subop        = Input(UInt(5.W))
    val start        = Input(Bool())
    val roundingMode = Input(UInt(3.W))
    val ready        = Output(Bool())
    val valid        = Output(Bool())
    val result       = Output(UInt(32.W))
    val flags        = Output(UInt(5.W))
  })

  val a   = recFNFromFN(8, 24, io.a)
  val b   = recFNFromFN(8, 24, io.b)
  val c   = recFNFromFN(8, 24, io.c)
  val add = Module(new AddRecFN(8, 24))
  add.io.a              := Mux(io.op === 39.U, b, a)
  add.io.b              := Mux(io.op === 39.U, a, b)
  add.io.subOp          := io.op === 2.U || io.op === 39.U
  add.io.roundingMode   := io.roundingMode
  add.io.detectTininess := 1.U
  val mul = Module(new MulRecFN(8, 24))
  mul.io.a              := a
  mul.io.b              := b
  mul.io.roundingMode   := io.roundingMode
  mul.io.detectTininess := 1.U
  val fma = Module(new MulAddRecFN(8, 24))
  fma.io.a              := b
  fma.io.b              := Mux(io.op(2), a, c)
  fma.io.c              := Mux(io.op(2), c, a)
  fma.io.op             := Cat(io.op(0), io.op(1) ^ io.op(0))
  fma.io.roundingMode   := io.roundingMode
  fma.io.detectTininess := 1.U
  val cmp = Module(new CompareRecFN(8, 24))
  cmp.io.a         := a
  cmp.io.b         := b
  cmp.io.signaling := io.op === 25.U || io.op === 27.U || io.op === 29.U || io.op === 31.U
  val squareRoot    = io.op === 19.U && io.subop === 0.U
  val longOperation = io.op === 32.U || io.op === 33.U || squareRoot
  val div           = Module(new DivSqrtRecFN_small(8, 24, 0))
  div.io.a              := Mux(io.op === 33.U, b, a)
  div.io.b              := Mux(io.op === 33.U, a, b)
  div.io.sqrtOp         := squareRoot
  div.io.roundingMode   := io.roundingMode
  div.io.detectTininess := 1.U
  div.io.inValid        := io.start && longOperation

  val aNaN     = io.a(30, 23).andR && io.a(22, 0).orR
  val aSNaN    = aNaN && !io.a(22)
  val bNaN     = io.b(30, 23).andR && io.b(22, 0).orR
  val aZero    = !io.a(30, 0).orR
  val aInf     = io.a(30, 23).andR && !io.a(22, 0).orR
  val bothZero = aZero && !io.b(30, 0).orR

  val minimum = Mux(
    aNaN && bNaN,
    "h7fc00000".U,
    Mux(aNaN, io.b, Mux(bNaN, io.a, Mux(bothZero, io.a | io.b, Mux(cmp.io.lt, io.a, io.b))))
  )

  val maximum = Mux(
    aNaN && bNaN,
    "h7fc00000".U,
    Mux(aNaN, io.b, Mux(bNaN, io.a, Mux(bothZero, io.a & io.b, Mux(cmp.io.gt, io.a, io.b))))
  )

  val toInt32 = Module(new RecFNToIN(8, 24, 32))
  val toInt16 = Module(new RecFNToIN(8, 24, 16))
  for (convert <- Seq(toInt32, toInt16)) {
    convert.io.in           := a
    convert.io.signedOut    := io.subop(0)
    convert.io.roundingMode := Mux(io.subop(2), 1.U, io.roundingMode)
  }
  val toFloat = Module(new INToRecFN(32, 8, 24))
  toFloat.io.in             := Mux(io.subop(3), Cat(Fill(16, io.subop(0) && io.a(15)), io.a(15, 0)), io.a)
  toFloat.io.signedIn       := io.subop(0)
  toFloat.io.roundingMode   := io.roundingMode
  toFloat.io.detectTininess := 1.U
  val intFlags  = Mux(io.subop(4), toInt16.io.intExceptionFlags, toInt32.io.intExceptionFlags)
  val intResult = Mux(io.subop(4), toInt16.io.out, toInt32.io.out)

  // RVV 1.0 vfrec7/vfrsqrt7 tables; indices use the normalized input fraction.
  val reciprocalTable = VecInit(Seq(
    127, 125, 123, 121, 119, 117, 116, 114, 112, 110, 109, 107, 105, 104, 102, 100, 99, 97, 96, 94, 93, 91, 90, 88, 87,
    85, 84, 83, 81, 80, 79, 77, 76, 75, 74, 72, 71, 70, 69, 68, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53,
    52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 40, 39, 38, 37, 36, 35, 35, 34, 33, 32, 31, 31, 30, 29, 28, 28,
    27, 26, 25, 25, 24, 23, 23, 22, 21, 21, 20, 19, 19, 18, 17, 17, 16, 15, 15, 14, 14, 13, 12, 12, 11, 11, 10, 9, 9, 8,
    8, 7, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0
  ).map(_.U(7.W)))

  val reciprocalSqrtTable = VecInit(Seq(
    52, 51, 50, 48, 47, 46, 44, 43, 42, 41, 40, 39, 38, 36, 35, 34, 33, 32, 31, 30, 30, 29, 28, 27, 26, 25, 24, 23, 23,
    22, 21, 20, 19, 19, 18, 17, 16, 16, 15, 14, 14, 13, 12, 12, 11, 10, 10, 9, 9, 8, 7, 7, 6, 6, 5, 4, 4, 3, 3, 2, 2, 1,
    1, 0, 127, 125, 123, 121, 119, 118, 116, 114, 113, 111, 109, 108, 106, 105, 103, 102, 100, 99, 97, 96, 95, 93, 92,
    91, 90, 88, 87, 86, 85, 84, 83, 82, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 70, 69, 68, 67, 66, 65, 64, 63, 63,
    62, 61, 60, 59, 59, 58, 57, 56, 56, 55, 54, 53
  ).map(_.U(7.W)))

  val leadingZeros       = PriorityEncoder(Reverse(io.a(22, 0)))
  val normalizedExponent = Mux(io.a(30, 23).orR, Cat(0.U(1.W), io.a(30, 23)).asSInt, -leadingZeros.zext)
  val normalizedFraction = Mux(io.a(30, 23).orR, io.a(22, 0), (io.a(22, 0) << (leadingZeros +& 1.U))(22, 0))
  val reciprocalExponent = 253.S(10.W) - normalizedExponent
  val reciprocalFraction = Cat(reciprocalTable(normalizedFraction(22, 16)), 0.U(16.W))

  val reciprocalFinite = Cat(
    io.a(31),
    Mux(reciprocalExponent <= 0.S, 0.U(8.W), reciprocalExponent.asUInt(7, 0)),
    Mux(
      reciprocalExponent <= 0.S,
      (Cat(1.U(1.W), reciprocalFraction) >> Mux(reciprocalExponent === 0.S, 1.U, 2.U))(22, 0),
      reciprocalFraction
    )
  )

  val reciprocalOverflow = reciprocalExponent > 254.S
  val overflowToFinite   = io.roundingMode === 1.U ||
    (io.roundingMode === 2.U && !io.a(31)) || (io.roundingMode === 3.U && io.a(31))
  val signedInfinity     = Cat(io.a(31), "h7f800000".U(31.W))

  val reciprocal = Mux(
    aNaN,
    "h7fc00000".U,
    Mux(
      aInf,
      Cat(io.a(31), 0.U(31.W)),
      Mux(
        aZero,
        signedInfinity,
        Mux(
          reciprocalOverflow,
          Mux(overflowToFinite, Cat(io.a(31), "h7f7fffff".U(31.W)), signedInfinity),
          reciprocalFinite
        )
      )
    )
  )

  val reciprocalFlags =
    Mux(aNaN, Cat(aSNaN, 0.U(4.W)), Mux(aInf, 0.U, Mux(aZero, 8.U, Mux(reciprocalOverflow, 5.U, 0.U))))
  val sqrtExponent    = ((380.S(10.W) - normalizedExponent) >> 1).asUInt
  val sqrtFraction    = reciprocalSqrtTable(Cat(normalizedExponent.asUInt(0), normalizedFraction(22, 17)))
  val sqrtInvalid     = aSNaN || (!aNaN && io.a(31) && !aZero)

  val reciprocalSqrt = Mux(
    aNaN || sqrtInvalid,
    "h7fc00000".U,
    Mux(aZero, signedInfinity, Mux(aInf, 0.U, Cat(io.a(31), sqrtExponent(7, 0), sqrtFraction, 0.U(16.W))))
  )

  val reciprocalSqrtFlags = Mux(sqrtInvalid, 16.U, Mux(aZero, 8.U, 0.U))

  io.ready  := div.io.inReady
  io.valid  := Mux(longOperation, div.io.outValid_div || div.io.outValid_sqrt, io.start)
  io.result := 0.U
  io.flags  := 0.U
  val supported = WireDefault(true.B)
  switch(io.op) {
    is(0.U, 1.U, 2.U, 3.U, 39.U) {
      io.result := fNFromRecFN(8, 24, add.io.out)
      io.flags  := add.io.exceptionFlags
    }
    is(4.U, 5.U) { io.result := minimum; io.flags := cmp.io.exceptionFlags }
    is(6.U, 7.U) { io.result := maximum; io.flags := cmp.io.exceptionFlags }
    is(8.U)(io.result  := Cat(io.b(31), io.a(30, 0)))
    is(9.U)(io.result  := Cat(!io.b(31), io.a(30, 0)))
    is(10.U)(io.result := Cat(io.a(31) ^ io.b(31), io.a(30, 0)))
    is(18.U) {
      supported := Seq(0, 1, 2, 3, 6, 7, 10, 11, 16, 17, 22, 23).map(x => io.subop === x.U).reduce(_ || _)
      when(io.subop === 2.U || io.subop === 3.U || io.subop === 10.U || io.subop === 11.U) {
        io.result := fNFromRecFN(8, 24, toFloat.io.out)
        io.flags  := toFloat.io.exceptionFlags
      }.otherwise {
        io.result := intResult
        io.flags  := Cat(intFlags(2, 1).orR, 0.U(3.W), intFlags(0))
      }
    }
    is(19.U) {
      supported := Seq(0, 4, 5, 16).map(x => io.subop === x.U).reduce(_ || _)
      switch(io.subop) {
        is(0.U) { io.result := fNFromRecFN(8, 24, div.io.out); io.flags := div.io.exceptionFlags }
        is(4.U) { io.result := reciprocalSqrt; io.flags := reciprocalSqrtFlags }
        is(5.U) { io.result := reciprocal; io.flags := reciprocalFlags }
        is(16.U)(io.result := classifyRecFN(8, 24, a))
      }
    }
    is(24.U) { io.result := cmp.io.eq; io.flags := cmp.io.exceptionFlags }
    is(25.U) { io.result := cmp.io.lt || cmp.io.eq; io.flags := cmp.io.exceptionFlags }
    is(27.U) { io.result := cmp.io.lt; io.flags := cmp.io.exceptionFlags }
    is(28.U) { io.result := !cmp.io.eq; io.flags := cmp.io.exceptionFlags }
    is(29.U) { io.result := cmp.io.gt; io.flags := cmp.io.exceptionFlags }
    is(31.U) { io.result := cmp.io.gt || cmp.io.eq; io.flags := cmp.io.exceptionFlags }
    is(32.U, 33.U) { io.result := fNFromRecFN(8, 24, div.io.out); io.flags := div.io.exceptionFlags }
    is(36.U) { io.result := fNFromRecFN(8, 24, mul.io.out); io.flags := mul.io.exceptionFlags }
    is(40.U, 41.U, 42.U, 43.U, 44.U, 45.U, 46.U, 47.U) {
      io.result := fNFromRecFN(8, 24, fma.io.out)
      io.flags  := fma.io.exceptionFlags
    }
  }
  when(io.start) {
    assert(
      supported && Seq(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 19, 24, 25, 27, 28, 29, 31, 32, 33, 36, 39, 40, 41, 42, 43,
        44, 45, 46, 47).map(x => io.op === x.U).reduce(_ || _),
      "unsupported Zve32f floating-point operation"
    )
  }
}
