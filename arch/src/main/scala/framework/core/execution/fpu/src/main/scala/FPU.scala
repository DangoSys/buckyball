package framework.core.execution.fpu

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.lib.arith.Fma
import framework.core.execution.fpu.configs.{Params => ModuleParam}

class Request(b: ModuleParam) extends Bundle {
  val p = b

  val instruction = UInt(32.W)
  val integer     = UInt(p.xLen.W)
  val a           = UInt(p.fLen.W)
  val rhs         = UInt(p.fLen.W)
  val c           = UInt(p.fLen.W)
  val rounding    = UInt(3.W)
}

class Response(b: ModuleParam) extends Bundle {
  val p = b

  val data    = UInt(p.xLen.W)
  val flags   = UInt(5.W)
  val integer = Bool()
  val illegal = Bool()
}

/** IEEE binary32/binary64 arithmetic, conversions and register transfers. */
class FPUIO(b: ModuleParam) extends Bundle {
  val p = b

  val request  = Flipped(Decoupled(new Request(b)))
  val response = Decoupled(new Response(b))

}

@instantiable
class FPU(b: ModuleParam) extends Module {
  val p = b

  @public
  val io = IO(new FPUIO(b))

  val idle :: arithmetic :: divide :: respond :: Nil = Enum(4)
  val state                                          = RegInit(idle)
  val selectedDouble                                 = Reg(Bool())
  val answer                                         = Reg(new Response(b))
  val request                                        = io.request.bits
  val instruction                                    = request.instruction
  val opcode                                         = instruction(6, 0)
  val operation                                      = instruction(31, 27)
  val funct3                                         = instruction(14, 12)
  val rs2                                            = instruction(24, 20)
  val fused                                          = opcode === "h43".U || opcode === "h47".U || opcode === "h4b".U || opcode === "h4f".U
  val double                                         = instruction(25)
  val rounding                                       = Mux(funct3 === 7.U, request.rounding, funct3)
  val rounded                                        = fused || operation <= 3.U || operation === 8.U || operation === 11.U ||
    operation === 24.U || operation === 26.U
  val formatLegal                                    = instruction(26) === 0.U
  val legal                                          = WireDefault(false.B)
  val toInteger                                      = !fused && (operation === 20.U || operation === 24.U || operation === 28.U)
  val isArithmetic                                   = fused || operation <= 2.U
  val isDivide                                       = operation === 3.U || operation === 11.U
  when(fused)(legal            := true.B)
  when(opcode === "h53".U) {
    switch(operation) {
      is(0.U, 1.U, 2.U, 3.U)(legal := true.B)
      is(4.U)(legal                := funct3 <= 2.U)
      is(5.U)(legal                := funct3 <= 1.U)
      is(8.U)(legal                := rs2 === Mux(double, 0.U, 1.U))
      is(11.U)(legal               := rs2 === 0.U)
      is(20.U)(legal               := funct3 <= 2.U)
      is(24.U, 26.U)(legal         := rs2 <= 3.U)
      is(28.U)(legal               := rs2 === 0.U && funct3 <= 1.U)
      is(30.U)(legal               := rs2 === 0.U && funct3 === 0.U)
    }
  }
  val illegal = !legal || !formatLegal || (rounded && rounding > 4.U)
  io.request.ready             := state === idle
  io.response.valid            := state === respond
  io.response.bits             := answer
  when(io.response.fire)(state := idle)

  val simpleData      = Wire(Vec(2, UInt(64.W)))
  val simpleFlags     = Wire(Vec(2, UInt(5.W)))
  val arithmeticData  = Wire(Vec(2, UInt(64.W)))
  val arithmeticFlags = Wire(Vec(2, UInt(5.W)))
  val arithmeticValid = Wire(Vec(2, Bool()))
  val divideData      = Wire(Vec(2, UInt(64.W)))
  val divideFlags     = Wire(Vec(2, UInt(5.W)))
  val divideValid     = Wire(Vec(2, Bool()))
  val divideReady     = Wire(Vec(2, Bool()))

  for (((exp, sig), index) <- Seq((8, 24), (11, 53)).zipWithIndex) {
    val width     = exp + sig
    val single    = index == 0
    val canonical = if (single) BigInt("7fc00000", 16) else BigInt("7ff8000000000000", 16)
    val rawA      = if (single) Mux(request.a(63, 32).andR, request.a(31, 0), canonical.U(32.W)) else request.a
    val rawB      = if (single) Mux(request.rhs(63, 32).andR, request.rhs(31, 0), canonical.U(32.W)) else request.rhs
    val rawC      = if (single) Mux(request.c(63, 32).andR, request.c(31, 0), canonical.U(32.W)) else request.c
    val a         = hardfloat.recFNFromFN(exp, sig, rawA)
    val b         = hardfloat.recFNFromFN(exp, sig, rawB)
    val c         = hardfloat.recFNFromFN(exp, sig, rawC)
    val unpackedA = hardfloat.rawFloatFromRecFN(exp, sig, a)
    val unpackedB = hardfloat.rawFloatFromRecFN(exp, sig, b)
    val signNaN   = hardfloat.isSigNaNRawFloat(unpackedA) || hardfloat.isSigNaNRawFloat(unpackedB)
    val start     = io.request.fire && !illegal && double === (index == 1).B
    val fma: Instance[Fma] = Instantiate(new Fma(3, exp, sig))
    fma.io.validin        := start && isArithmetic
    fma.io.roundingMode   := rounding
    fma.io.detectTininess := 1.U
    fma.io.op             := Mux(fused, opcode(3, 2), Mux(operation === 1.U, 1.U, 0.U))
    fma.io.a              := a
    fma.io.b              := Mux(
      fused || operation === 2.U,
      b,
      hardfloat.recFNFromFN(exp, sig, (BigInt((1 << (exp - 1)) - 1) << (sig - 1)).U(width.W))
    )
    fma.io.c              := Mux(
      fused,
      c,
      Mux(
        operation === 2.U,
        hardfloat.recFNFromFN(exp, sig, Cat(rawA(width - 1) ^ rawB(width - 1), 0.U((width - 1).W))),
        b
      )
    )
    val fmaNan = hardfloat.rawFloatFromRecFN(exp, sig, fma.io.out).isNaN
    val fmaRaw = Mux(fmaNan, canonical.U(width.W), hardfloat.fNFromRecFN(exp, sig, fma.io.out))
    arithmeticData(index)  := (if (single) Cat(~0.U(32.W), fmaRaw) else fmaRaw)
    arithmeticFlags(index) := fma.io.exceptionFlags
    arithmeticValid(index) := fma.io.validout

    val div = Module(new hardfloat.DivSqrtRecFN_small(exp, sig, 0))
    div.io.inValid        := start && isDivide
    div.io.sqrtOp         := operation === 11.U
    div.io.a              := a
    div.io.b              := b
    div.io.roundingMode   := rounding
    div.io.detectTininess := 1.U
    val divRaw = Mux(
      hardfloat.rawFloatFromRecFN(exp, sig, div.io.out).isNaN,
      canonical.U(width.W),
      hardfloat.fNFromRecFN(exp, sig, div.io.out)
    )
    divideData(index)  := (if (single) Cat(~0.U(32.W), divRaw) else divRaw)
    divideFlags(index) := div.io.exceptionFlags
    divideValid(index) := div.io.outValid_div || div.io.outValid_sqrt
    divideReady(index) := div.io.inReady

    val compare = Module(new hardfloat.CompareRecFN(exp, sig))
    compare.io.a         := a
    compare.io.b         := b
    compare.io.signaling := funct3 =/= 2.U
    val bothZero       = unpackedA.isZero && unpackedB.isZero
    val chooseA        =
      Mux(funct3(0), compare.io.gt || (bothZero && !rawA(width - 1)), compare.io.lt || (bothZero && rawA(width - 1)))
    val minimumMaximum = Mux(
      unpackedA.isNaN && unpackedB.isNaN,
      canonical.U(width.W),
      Mux(unpackedA.isNaN, rawB, Mux(unpackedB.isNaN, rawA, Mux(chooseA, rawA, rawB)))
    )
    val injectedSign   = MuxLookup(funct3, rawB(width - 1))(Seq(
      1.U -> !rawB(width - 1),
      2.U -> (rawA(width - 1) ^ rawB(width - 1))
    ))
    val injected       = Cat(injectedSign, rawA(width - 2, 0))

    val intInput = Mux(rs2(1), request.integer, Cat(Fill(32, !rs2(0) && request.integer(31)), request.integer(31, 0)))
    val fromInt  = Module(new hardfloat.INToRecFN(64, exp, sig))
    fromInt.io.in             := intInput
    fromInt.io.signedIn       := !rs2(0)
    fromInt.io.roundingMode   := rounding
    fromInt.io.detectTininess := 1.U
    val fromIntRaw        = hardfloat.fNFromRecFN(exp, sig, fromInt.io.out)
    val convertedIntegers = Wire(Vec(2, UInt(64.W)))
    val convertedFlags    = Wire(Vec(2, UInt(5.W)))
    for ((intWidth, intIndex) <- Seq(32, 64).zipWithIndex) {
      val toInt = Module(new hardfloat.RecFNToIN(exp, sig, intWidth))
      toInt.io.in                 := a
      toInt.io.signedOut          := !rs2(0)
      toInt.io.roundingMode       := rounding
      convertedIntegers(intIndex) := (if (intWidth == 32) Cat(Fill(32, toInt.io.out(31)), toInt.io.out)
                                      else toInt.io.out)
      convertedFlags(intIndex)    := Cat(toInt.io.intExceptionFlags(2, 1).orR, 0.U(3.W), toInt.io.intExceptionFlags(0))
    }
    val (sourceExp, sourceSig) = if (single) (11, 53) else (8, 24)
    val source  =
      if (single) request.a
      else
        Mux(request.a(63, 32).andR, request.a(31, 0), "h7fc00000".U(32.W))
    val convert = Module(new hardfloat.RecFNToRecFN(sourceExp, sourceSig, exp, sig))
    convert.io.in             := hardfloat.recFNFromFN(sourceExp, sourceSig, source)
    convert.io.roundingMode   := rounding
    convert.io.detectTininess := 1.U
    val convertedFloat = Mux(
      hardfloat.rawFloatFromRecFN(exp, sig, convert.io.out).isNaN,
      canonical.U(width.W),
      hardfloat.fNFromRecFN(exp, sig, convert.io.out)
    )
    val floatResult    = MuxLookup(operation, 0.U(width.W))(Seq(
      4.U  -> injected,
      5.U  -> minimumMaximum,
      8.U  -> convertedFloat,
      26.U -> fromIntRaw,
      30.U -> request.integer(width - 1, 0)
    ))
    val integerResult  = MuxLookup(operation, 0.U(64.W))(Seq(
      20.U -> Cat(
        0.U(63.W),
        Mux(funct3 === 0.U, compare.io.lt || compare.io.eq, Mux(funct3 === 1.U, compare.io.lt, compare.io.eq))
      ),
      24.U -> convertedIntegers(rs2(1)),
      28.U -> Mux(
        funct3(0),
        hardfloat.classifyRecFN(exp, sig, a).pad(64),
        if (single) Cat(Fill(32, request.a(31)), request.a(31, 0)) else request.a
      )
    ))
    simpleData(index) := Mux(toInteger, integerResult, if (single) Cat(~0.U(32.W), floatResult) else floatResult)
    simpleFlags(index) := MuxLookup(operation, 0.U(5.W))(Seq(
      5.U  -> Cat(signNaN, 0.U(4.W)),
      8.U  -> convert.io.exceptionFlags,
      20.U -> compare.io.exceptionFlags,
      24.U -> convertedFlags(rs2(1)),
      26.U -> fromInt.io.exceptionFlags
    ))
  }

  when(io.request.fire) {
    selectedDouble := double
    answer.data    := simpleData(double)
    answer.flags   := Mux(illegal, 0.U, simpleFlags(double))
    answer.integer := toInteger
    answer.illegal := illegal
    state          := Mux(illegal, respond, Mux(isArithmetic, arithmetic, Mux(isDivide, divide, respond)))
    when(!illegal && isDivide)(assert(divideReady(double), "Floating divider must be idle on accepted command"))
  }
  when(state === arithmetic && arithmeticValid(selectedDouble)) {
    answer.data  := arithmeticData(selectedDouble)
    answer.flags := arithmeticFlags(selectedDouble)
    state        := respond
  }
  when(state === divide && divideValid(selectedDouble)) {
    answer.data  := divideData(selectedDouble)
    answer.flags := divideFlags(selectedDouble)
    state        := respond
  }
}
