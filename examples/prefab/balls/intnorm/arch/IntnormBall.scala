package examples.balls.intnorm

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam
import framework.ball.sub_rob.ExpansionParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.ball.blink.{BlinkIO, HasBlink}
import framework.ball.bbus.{BallFactory, BallPorts}
import framework.generic.arith.Multiply
import buckyball.balls.intnorm.{Params => ParametersProto}

@instantiable
class IntnormBall(
  link:  BlinkParam,
  route: BallRouteParam)
    extends Module
    with HasBlink {
  val ball = route
  ball.parameters.unpack(classOf[ParametersProto])
  require(ball.instructions.forall(i => (i.funct7 >> 4) == 4), "INTNORM requires a two-read, one-write opcode")
  require(ball.inBW == 2 && ball.outBW == 1)
  require(link.bankWidth == 128 && link.bankMaskLen == 16 && link.dataBits == 64)

  @public val blink = IO(new BlinkIO(link, route))

  val (idle :: headerRead :: headerWait :: inputRead :: inputWait :: sumValues ::
    gammaRead :: gammaWait :: betaRead :: betaWait :: multiplyStart :: multiplyWait ::
    divide :: squareRoot :: scratchRead :: scratchWait :: encode :: writing :: acknowledge :: complete :: Nil) = Enum(20)

  val state            = RegInit(idle)
  val robId            = Reg(UInt(link.robIdBits.W))
  val source           = Reg(UInt(link.bankIdBits.W))
  val coefficients     = Reg(UInt(link.bankIdBits.W))
  val destination      = Reg(UInt(link.bankIdBits.W))
  val sourceGroup      = Reg(UInt(link.groupIdBits.W))
  val coefficientGroup = Reg(UInt(link.groupIdBits.W))
  val targetGroup      = Reg(UInt(link.groupIdBits.W))
  val rows             = Reg(UInt(link.iterBits.W))
  val k                = Reg(UInt(16.W))
  val stride           = Reg(UInt(16.W))
  val inputBase        = Reg(UInt(32.W))
  val coefficientBase  = Reg(UInt(32.W))
  val outputBase       = Reg(UInt(32.W))
  val scratchBase      = Reg(UInt(32.W))
  val row              = Reg(UInt(link.iterBits.W))
  val column           = Reg(UInt(16.W))
  val phase            = Reg(UInt(2.W)) // Sum, variance, affine.
  val values           = Reg(Vec(4, SInt(33.W)))
  val gamma            = Reg(Vec(4, SInt(32.W)))
  val beta             = Reg(Vec(4, SInt(32.W)))
  val normalized       = Reg(Vec(4, SInt(32.W)))
  val seeds            = Reg(Vec(4, UInt(32.W)))
  val sum              = Reg(SInt(64.W))
  val mean             = Reg(SInt(32.W))
  val squares          = Reg(UInt(80.W))
  val epsilon          = Reg(UInt(64.W))
  val root             = Reg(UInt(32.W))
  val reciprocal       = Reg(UInt(64.W))
  val multiplyKind     = Reg(UInt(2.W)) // Square, normalize, quotient correction, affine.
  val multipliers      = Seq.fill(4)(Seq.fill(2)(Instantiate(new Multiply(16))))

  val quotient      = Reg(UInt(80.W))
  val divisor       = Reg(UInt(64.W))
  val remainder     = Reg(UInt(65.W))
  val divideStep    = Reg(UInt(7.W))
  val divideKind    = Reg(UInt(2.W)) // Mean, variance, reciprocal.
  val negative      = Reg(Bool())
  val sqrtInput     = Reg(UInt(64.W))
  val sqrtRoot      = Reg(UInt(32.W))
  val sqrtRemainder = Reg(UInt(34.W))
  val sqrtStep      = Reg(UInt(5.W))

  val block        = Reg(UInt(16.W))
  val chunk        = Reg(UInt(3.W))
  val blockValues  = Reg(Vec(32, SInt(32.W)))
  val maximum      = Reg(UInt(32.W))
  val scaleRow     = Reg(Vec(16, UInt(8.W)))
  val secondWord   = Reg(UInt(128.W))
  val writeWord    = Reg(UInt(128.W))
  val writeAddress = Reg(UInt(32.W))
  val writeKind    = Reg(UInt(2.W)) // Scratch, first codes, second codes, exponents.

  // Decode the source bit pattern with integer shifts and RNE.
  def fixed(bits: UInt): SInt = {
    val e         = bits(30, 23)
    val mantissa  = Cat(1.U(1.W), bits(22, 0)).pad(33)
    val right     = (134.U - e)(4, 0)
    val shifted   = mantissa >> right
    val mask      = ((1.U(33.W) << right) - 1.U)(32, 0)
    val rest      = mantissa & mask
    val half      = (1.U(33.W) << ((right - 1.U)(4, 0)))(32, 0)
    val increment = right =/= 0.U && (rest > half || rest === half && shifted(0))
    val magnitude =
      Mux(e >= 134.U, (mantissa << ((e - 134.U)(3, 0)))(32, 0), Mux(e < 110.U, 0.U(33.W), shifted + increment.asUInt))
    assert(e =/= 255.U && e <= 142.U, "INTNORM input must be finite and fit signed Q16")
    assert(
      magnitude <= Mux(bits(31), (BigInt(1) << 31).U(33.W), ((BigInt(1) << 31) - 1).U(33.W)),
      "INTNORM input must fit signed Q16"
    )
    Mux(bits(31), -magnitude(31, 0), magnitude(31, 0)).asSInt
  }

  def round16(bits: UInt): SInt = {
    val magnitude = Mux(bits(63), -bits, bits)
    val shifted   = magnitude >> 16
    val increment = magnitude(15, 0) > 32768.U || magnitude(15, 0) === 32768.U && shifted(0)
    val rounded   = shifted + increment.asUInt
    Mux(bits(63), -rounded, rounded).asSInt
  }

  def code(value: SInt, bits: UInt): UInt = {
    val raw       = value.asUInt
    val magnitude = Mux(raw(31), -raw, raw)
    val right     = (bits - 7.U)(4, 0)
    val shifted   = magnitude >> right
    val mask      = ((1.U(33.W) << right) - 1.U)(31, 0)
    val rest      = magnitude & mask
    val half      = (1.U(32.W) << ((right - 1.U)(4, 0)))(31, 0)
    val increment = right =/= 0.U && (rest > half || rest === half && shifted(0))
    val rounded   = Mux(bits >= 7.U, (shifted + increment.asUInt).pad(39), magnitude << ((7.U - bits)(2, 0)))
    val clipped   = Mux(rounded > 127.U, 127.U(8.W), rounded(7, 0))
    Mux(raw(31), -clipped, clipped)
  }

  val active     = !reset.asBool
  val lastColumn = column === (k >> 2) - 1.U
  val rowBase    = outputBase + row * stride
  val centered   = values.map(v => v - mean.pad(33))
  val magnitudes = centered.map(v => Mux(v < 0.S, -v, v).asUInt.pad(64))
  blink.cmdReq.ready        := state === idle && active
  blink.cmdResp.valid       := state === complete && active
  blink.cmdResp.bits.rob_id := robId
  blink.status.idle         := state === idle
  blink.status.running      := state =/= idle && state =/= complete

  val read       = blink.bankRead(0)
  val packedRead = state === scratchRead || state === scratchWait
  read.rob_id           := robId
  read.bank_id          := Mux(packedRead, destination, source)
  read.group_id         := Mux(packedRead, targetGroup, sourceGroup)
  read.io.req.valid     := (state === inputRead || state === scratchRead) && active
  read.io.req.bits.addr := Mux(packedRead, scratchBase + (block << 3) + chunk, inputBase + row * (k >> 2) + column)
  read.io.resp.ready    := (state === inputWait || state === scratchWait) && active

  val coefficientRead = blink.bankRead(1)
  coefficientRead.rob_id           := robId
  coefficientRead.bank_id          := coefficients
  coefficientRead.group_id         := coefficientGroup
  coefficientRead.io.req.valid     := (state === headerRead || state === gammaRead || state === betaRead) && active
  coefficientRead.io.req.bits.addr := coefficientBase + Mux(
    state === headerRead,
    0.U,
    1.U + column + Mux(state === betaRead, k >> 2, 0.U)
  )
  coefficientRead.io.resp.ready    := (state === headerWait || state === gammaWait || state === betaWait) && active

  val write = blink.bankWrite(0)
  write.rob_id           := robId
  write.bank_id          := destination
  write.group_id         := targetGroup
  write.io.req.valid     := state === writing && active
  write.io.req.bits.addr := writeAddress
  write.io.req.bits.data := writeWord
  write.io.req.bits.mask := VecInit(Seq.fill(16)(true.B))
  write.io.resp.ready    := state === acknowledge && active

  val multiplyReady = multipliers.flatten.map(_.io.request.ready).reduce(_ && _)
  val multiplyValid = multipliers.flatten.map(_.io.response.valid).reduce(_ && _)
  val products      = multipliers.map(pair => pair(0).io.response.bits + (pair(1).io.response.bits << 16)(63, 0))
  for (i <- 0 until 4) {
    // Centered Q16 magnitudes, roots and affine coefficients fit 32 bits.
    val a = VecInit(Seq(magnitudes(i), reciprocal, seeds(i).pad(64), normalized(i).pad(64).asUInt))
    val b = VecInit(Seq(magnitudes(i), magnitudes(i), root.pad(64), gamma(i).pad(64).asUInt))
    // Unsigned low half plus a signed high half preserves the full product.
    for (part <- 0 until 2) {
      val multiplier = multipliers(i)(part)
      multiplier.io.request.valid        := state === multiplyStart && multiplyReady && active
      multiplier.io.request.bits.a       := a(multiplyKind)
      multiplier.io.request.bits.b       := (if (part == 0) b(multiplyKind)(15, 0).pad(64)
                                       else Mux(
                                         multiplyKind === 3.U,
                                         b(multiplyKind)(31, 16).asSInt.pad(64).asUInt,
                                         b(multiplyKind)(31, 16).pad(64)
                                       ))
      multiplier.io.request.bits.signedA := multiplyKind === 3.U
      multiplier.io.request.bits.signedB := (if (part == 0) false.B else multiplyKind === 3.U)
      multiplier.io.request.bits.high    := false.B
      multiplier.io.response.ready       := state === multiplyWait && multiplyValid && active
    }
  }

  when(state === idle && blink.cmdReq.fire) {
    val cmd    = blink.cmdReq.bits
    val count  = cmd.rs2(15, 0)
    val in     = cmd.rs2(29, 16)
    val out    = cmd.rs2(43, 30)
    val coeff  = cmd.rs2(57, 44)
    val packed = (count >> 4) +& (count >> 9)
    assert(
      cmd.rs2(63, 58) === 0.U && count =/= 0.U && count(8, 0) === 0.U,
      "INTNORM requires a positive multiple of 512 and zero reserved fields"
    )
    assert(cmd.iter =/= 0.U, "INTNORM rows must be positive")
    assert(
      cmd.op1_bank =/= cmd.op2_bank && cmd.op1_bank =/= cmd.wr_bank && cmd.op2_bank =/= cmd.wr_bank,
      "INTNORM requires three distinct banks"
    )
    assert(cmd.op1_col === 1.U && cmd.op2_col === 1.U && cmd.wr_col === 1.U, "INTNORM banks must have one group")
    assert(in +& cmd.iter * (count >> 2) <= link.bankEntries.U, "INTNORM input exceeds capacity")
    assert(coeff +& 1.U +& (count >> 1) <= link.bankEntries.U, "INTNORM coefficients exceed capacity")
    assert(out +& cmd.iter * packed +& (count >> 2) <= link.bankEntries.U, "INTNORM output and scratch exceed capacity")
    robId            := cmd.rob_id
    source           := cmd.op1_bank
    coefficients     := cmd.op2_bank
    destination      := cmd.wr_bank
    sourceGroup      := cmd.op1_group
    coefficientGroup := cmd.op2_group
    targetGroup      := cmd.wr_group
    rows             := cmd.iter
    k                := count
    stride           := packed
    inputBase        := in
    coefficientBase  := coeff
    outputBase       := out
    scratchBase      := out +& cmd.iter * packed
    row              := 0.U
    column           := 0.U
    phase            := 0.U
    sum              := 0.S
    state            := headerRead
  }
  when(state === headerRead && coefficientRead.io.req.fire)(state := headerWait)
  when(state === headerWait && coefficientRead.io.resp.fire) {
    assert(coefficientRead.io.resp.bits.data(63, 0) =/= 0.U, "INTNORM epsilon must be positive")
    assert(coefficientRead.io.resp.bits.data(127, 64) === 0.U, "INTNORM reserved coefficient words must be zero")
    epsilon := coefficientRead.io.resp.bits.data(63, 0)
    state   := inputRead
  }
  when(state === inputRead && read.io.req.fire)(state             := inputWait)
  when(state === inputWait && read.io.resp.fire) {
    for (i <- 0 until 4)
      values(i) := fixed(read.io.resp.bits.data(32 * i + 31, 32 * i))
    switch(phase) {
      is(0.U)(state := sumValues)
      is(1.U) { multiplyKind := 0.U; state := multiplyStart }
      is(2.U)(state := gammaRead)
    }
  }
  when(state === sumValues) {
    val next = sum + values.map(_.pad(64)).reduce(_ + _)
    sum := next
    when(lastColumn) {
      quotient   := Mux(next < 0.S, -next, next).asUInt.pad(80)
      divisor    := k
      remainder  := 0.U
      divideStep := 79.U
      divideKind := 0.U
      negative   := next < 0.S
      state      := divide
    }.otherwise { column := column + 1.U; state := inputRead }
  }
  when(state === gammaRead && coefficientRead.io.req.fire)(state  := gammaWait)
  when(state === gammaWait && coefficientRead.io.resp.fire) {
    for (i <- 0 until 4)
      gamma(i) := coefficientRead.io.resp.bits.data(32 * i + 31, 32 * i).asSInt
    state      := betaRead
  }
  when(state === betaRead && coefficientRead.io.req.fire)(state   := betaWait)
  when(state === betaWait && coefficientRead.io.resp.fire) {
    for (i <- 0 until 4)
      beta(i)    := coefficientRead.io.resp.bits.data(32 * i + 31, 32 * i).asSInt
    multiplyKind := 1.U
    state        := multiplyStart
  }
  when(state === multiplyStart && multiplyReady)(state            := multiplyWait)
  when(state === multiplyWait && multiplyValid) {
    switch(multiplyKind) {
      is(0.U) {
        val next = squares + products.map(_.pad(80)).reduce(_ + _)
        squares := next
        when(lastColumn) {
          quotient   := next
          divisor    := k
          remainder  := 0.U
          divideStep := 79.U
          divideKind := 1.U
          negative   := false.B
          state      := divide
        }.otherwise { column := column + 1.U; state := inputRead }
      }
      is(1.U) {
        for (i <- 0 until 4)
          seeds(i)   := products(i)(63, 32)
        multiplyKind := 2.U
        state        := multiplyStart
      }
      is(2.U) {
        for (i <- 0 until 4) {
          val rest      = (magnitudes(i) << 16)(63, 0) - products(i)
          val correct   = rest >= root
          val adjusted  = rest - Mux(correct, root.pad(64), 0.U(64.W))
          val q         = seeds(i).pad(64) + correct.asUInt
          val twice     = adjusted << 1
          val increment = twice > root || twice === root && q(0)
          val rounded   = q + increment.asUInt
          val signed    = Mux(centered(i) < 0.S, -rounded, rounded)
          normalized(i) := signed(31, 0).asSInt
        }
        multiplyKind := 3.U
        state := multiplyStart
      }
      is(3.U) {
        val results = (0 until 4).map { i =>
          val value = round16(products(i)) + beta(i).pad(64)
          assert(value === value(31, 0).asSInt.pad(64), "INTNORM result must fit signed Q16")
          value(31, 0)
        }
        writeWord := Cat(results.reverse)
        writeAddress := scratchBase + column
        writeKind    := 0.U
        state        := writing
      }
    }
  }

  when(state === divide) {
    val trial         = Cat(remainder(63, 0), quotient(79))
    val subtract      = trial >= divisor
    val nextRemainder = Mux(subtract, trial - divisor, trial)
    val nextQuotient  = Cat(quotient(78, 0), subtract)
    quotient   := nextQuotient
    remainder  := nextRemainder
    divideStep := divideStep - 1.U
    when(divideStep === 0.U) {
      val twice     = nextRemainder << 1
      val increment = twice > divisor || twice === divisor && nextQuotient(0)
      val rounded   = nextQuotient + increment.asUInt
      switch(divideKind) {
        is(0.U) {
          val signed = Mux(negative, -rounded(63, 0), rounded(63, 0))
          mean    := signed(31, 0).asSInt
          column  := 0.U
          squares := 0.U
          phase   := 1.U
          state   := inputRead
        }
        is(1.U) {
          val variance = rounded(63, 0) +& epsilon
          assert(!variance(64), "INTNORM variance exceeds u64")
          sqrtInput     := variance(63, 0)
          sqrtRoot      := 0.U
          sqrtRemainder := 0.U
          sqrtStep      := 0.U
          state         := squareRoot
        }
        is(2.U) {
          reciprocal := nextQuotient(63, 0)
          phase      := 2.U
          column     := 0.U
          state      := inputRead
        }
      }
    }
  }
  when(state === squareRoot) {
    val trial      = Cat(sqrtRemainder(31, 0), sqrtInput(63, 62))
    val subtractor = Cat(sqrtRoot, 0.U(1.W), 1.U(1.W))
    val subtract   = trial >= subtractor
    val next       = Cat(sqrtRoot(30, 0), subtract)
    sqrtRoot      := next
    sqrtRemainder := Mux(subtract, trial - subtractor, trial)
    sqrtInput     := sqrtInput << 2
    sqrtStep      := sqrtStep + 1.U
    when(sqrtStep === 31.U) {
      root       := next
      quotient   := (BigInt(1) << 48).U(80.W)
      divisor    := next
      remainder  := 0.U
      divideStep := 79.U
      divideKind := 2.U
      negative   := false.B
      state      := divide
    }
  }

  when(state === scratchRead && read.io.req.fire)(state := scratchWait)
  when(state === scratchWait && read.io.resp.fire) {
    val words = (0 until 4).map(i => read.io.resp.bits.data(32 * i + 31, 32 * i))
    for (i <- 0 until 4)
      blockValues(Cat(chunk, i.U(2.W))) := words(i).asSInt
    maximum                             := (Seq(maximum) ++ words.map(w => Mux(w(31), -w, w))).reduce((a, b) => Mux(a > b, a, b))
    when(chunk === 7.U)(state := encode)
      .otherwise { chunk := chunk + 1.U; state := scratchRead }
  }
  when(state === encode) {
    val bits   = Mux(maximum === 0.U, 0.U(6.W), 32.U(6.W) - PriorityEncoder(Reverse(maximum)))
    val packed = Cat(blockValues.reverse.map(v => code(v, bits)))
    scaleRow(block(3, 0)) := Mux(maximum === 0.U, 127.U(8.W), bits.pad(8) + 104.U(8.W))
    secondWord            := packed(255, 128)
    writeWord             := packed(127, 0)
    writeAddress          := rowBase + (block << 1)
    writeKind             := 1.U
    state                 := writing
  }
  when(state === writing && write.io.req.fire)(state    := acknowledge)
  when(state === acknowledge && write.io.resp.fire) {
    assert(write.io.resp.bits.ok, "INTNORM bank write failed")
    when(writeKind === 0.U) {
      when(lastColumn) {
        block   := 0.U
        chunk   := 0.U
        maximum := 0.U
        state   := scratchRead
      }.otherwise { column := column + 1.U; state := inputRead }
    }.elsewhen(writeKind === 1.U) {
      writeWord    := secondWord
      writeAddress := writeAddress + 1.U
      writeKind    := 2.U
      state        := writing
    }.elsewhen(writeKind === 2.U && block(3, 0) === 15.U) {
      writeWord    := Cat(scaleRow.reverse)
      writeAddress := rowBase + (k >> 4) + (block >> 4)
      writeKind    := 3.U
      state        := writing
    }.otherwise {
      when(block === (k >> 5) - 1.U) {
        when(row === rows - 1.U)(state := complete)
          .otherwise {
            row    := row + 1.U
            column := 0.U
            phase  := 0.U
            sum    := 0.S
            state  := inputRead
          }
      }.otherwise {
        block   := block + 1.U
        chunk   := 0.U
        maximum := 0.U
        state   := scratchRead
      }
    }
  }
  when(state === complete && blink.cmdResp.fire)(state  := idle)
}

object IntnormBall extends BallFactory {

  def instantiate(
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts =
    BallPorts(Instantiate(new IntnormBall(link, route)).blink, None)

}
