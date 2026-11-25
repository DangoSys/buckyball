package examples.balls.fixed

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam
import framework.ball.sub_rob.ExpansionParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.ball.blink.{BlinkIO, HasBlink}
import framework.ball.bbus.{BallFactory, BallPorts}
import framework.generic.arith.Multiply
import buckyball.balls.fixed.{Params => ParametersProto}
import framework.generic.contract.blink.BallCommand

@instantiable
class FixedBall(
  link:  BlinkParam,
  route: BallRouteParam)
    extends Module
    with HasBlink {
  val ball = route
  ball.parameters.unpack(classOf[ParametersProto])
  require(ball.inBW == 2 && ball.outBW == 1)
  require(link.bankWidth == 128 && link.bankMaskLen == 16 && link.bankEntries >= 8)

  @public val blink                                                                                                                                                                                                                                               = IO(new BlinkIO(link, route))
  val idle :: coefficientRequest :: coefficientResponse :: reading :: prepare :: scaleWait :: activate :: productWait :: divide :: pack :: writing :: acknowledge :: wideRequest :: wideResponse :: sumAccumulate :: meanPrepare :: meanDivide :: complete :: Nil =
    Enum(18)
  val state                                                                                                                                                                                                                                                       = RegInit(idle)
  val issue                                                                                                                                                                                                                                                       = Reg(new BallCommand(link))
  val mode                                                                                                                                                                                                                                                        = issue.rs2(2, 0)
  val broadcast                                                                                                                                                                                                                                                   = issue.rs2(3)
  val fraction                                                                                                                                                                                                                                                    = issue.rs2(8, 4)
  val stream                                                                                                                                                                                                                                                      = issue.rs2(9)
  val unit                                                                                                                                                                                                                                                        = (1.U(32.W) << fraction)(31, 0)
  val threshold                                                                                                                                                                                                                                                   = (3.U(32.W) << fraction)(31, 0)
  val row                                                                                                                                                                                                                                                         = RegInit(0.U(log2Ceil(link.bankEntries).W))
  val lane                                                                                                                                                                                                                                                        = RegInit(0.U(2.W))
  val coefficientRow                                                                                                                                                                                                                                              = RegInit(0.U(3.W))
  val multipliers                                                                                                                                                                                                                                                 = Reg(Vec(16, UInt(31.W)))
  val shifts                                                                                                                                                                                                                                                      = Reg(Vec(16, UInt(6.W)))
  val requested                                                                                                                                                                                                                                                   = RegInit(VecInit(Seq.fill(2)(false.B)))
  val received                                                                                                                                                                                                                                                    = RegInit(VecInit(Seq.fill(2)(false.B)))
  val words                                                                                                                                                                                                                                                       = Reg(Vec(2, UInt(128.W)))
  val output                                                                                                                                                                                                                                                      = Reg(Vec(4, UInt(32.W)))
  val scaled                                                                                                                                                                                                                                                      = Reg(UInt(32.W))
  val result                                                                                                                                                                                                                                                      = Reg(UInt(32.W))
  val channel                                                                                                                                                                                                                                                     = Mux(stream, lane.pad(4), Cat(row(1, 0), lane))
  val a                                                                                                                                                                                                                                                           = (words(0) >> (lane << 5))(31, 0)
  val b                                                                                                                                                                                                                                                           = (words(1) >> (lane << 5))(31, 0)
  val binary                                                                                                                                                                                                                                                      = mode === 3.U || mode === 5.U
  val multiply                                                                                                                                                                                                                                                    = Instantiate(new Multiply(32))
  val requantMultipliers                                                                                                                                                                                                                                          = Seq.fill(3)(Instantiate(new Multiply(32)))

  val sums          = Reg(Vec(16, SInt(64.W)))
  val divisors      = Reg(Vec(16, UInt(32.W)))
  val meanResults   = Reg(Vec(16, UInt(32.W)))
  val wideRow       = Reg(UInt(4.W))
  val meanChannel   = Reg(UInt(4.W))
  val meanQuotient  = Reg(UInt(64.W))
  val meanRemainder = Reg(UInt(32.W))
  val meanStep      = Reg(UInt(6.W))
  val meanNegative  = Reg(Bool())

  def clamp(value: SInt): UInt = {
    val bits = value.asUInt
    Mux(
      value > 2147483647L.S(64.W),
      "h7fffffff".U(32.W),
      Mux(value < -2147483648L.S(64.W), "h80000000".U(32.W), bits(31, 0))
    )
  }

  def roundShift(value: UInt, amount: UInt): UInt = {
    val negative  = value(63)
    val magnitude = Mux(negative, -value, value)
    val quotient  = magnitude >> amount
    val mask      = ((1.U(65.W) << amount) - 1.U)(63, 0)
    val remainder = magnitude & mask
    val half      = (1.U(64.W) << (amount - 1.U))(63, 0)
    val increment = remainder > half || (remainder === half && quotient(0))
    val rounded   = Mux(amount === 0.U, magnitude, quotient + increment.asUInt)
    clamp(Mux(negative, -rounded, rounded).asSInt)
  }

  // Divide the high part by three, seven bits per cycle. The low part is
  // retained so the final rounding is for the entire divisor 3 * 2^shift.
  val quotient       = Reg(UInt(28.W))
  val remainder      = Reg(UInt(2.W))
  val low            = Reg(UInt(25.W))
  val divideShift    = Reg(UInt(5.W))
  val divideNegative = Reg(Bool())
  val divideStep     = Reg(UInt(4.W))

  blink.cmdReq.ready        := state === idle
  blink.cmdResp.valid       := state === complete
  blink.cmdResp.bits.rob_id := issue.rob_id
  blink.status.idle         := state === idle
  blink.status.running      := state =/= idle
  for (port <- 0 until 2) {
    val read     = blink.bankRead(port)
    val metadata = if (port == 1) state === coefficientRequest || state === coefficientResponse else false.B
    val active   = if (port == 0) true.B else binary
    read.rob_id                                                 := issue.rob_id
    read.bank_id                                                := (if (port == 0) issue.op1_bank else issue.op2_bank)
    read.group_id                                               := (if (port == 0) issue.op1_group else issue.op2_group)
    read.io.req.bits.addr                                       := Mux(
      metadata,
      Mux(stream, (row << 1) + coefficientRow(0), coefficientRow),
      if (port == 0) row else Mux(broadcast, row(1, 0), row)
    )
    read.io.req.valid                                           := (state === reading && active && !requested(port)) ||
      (if (port == 1) state === coefficientRequest else false.B)
    read.io.resp.ready                                          := (state === reading && active && requested(port) && !received(port)) ||
      (if (port == 1) state === coefficientResponse else false.B)
    when(state === wideRequest || state === wideResponse) {
      val selected = if (port == 0) mode === 7.U && wideRow < 8.U else mode === 6.U || wideRow >= 8.U
      read.io.req.bits.addr        := Mux(mode === 7.U && wideRow >= 8.U, wideRow - 8.U, wideRow)
      read.io.req.valid            := selected && state === wideRequest
      read.io.resp.ready           := selected && state === wideResponse
      when(read.io.req.fire)(state := wideResponse)
      when(read.io.resp.fire) {
        when(mode === 6.U || wideRow < 8.U) {
          sums(Cat(wideRow(2, 0), 0.U(1.W))) := read.io.resp.bits.data(63, 0).asSInt
          sums(Cat(wideRow(2, 0), 1.U(1.W))) := read.io.resp.bits.data(127, 64).asSInt
        }.otherwise {
          for (i <- 0 until 4) {
            val value = read.io.resp.bits.data(i * 32 + 31, i * 32)
            assert(value =/= 0.U && !value(31), "FIXED mean divisor must be positive")
            divisors(Cat(wideRow(1, 0), i.U(2.W))) := value
          }
        }
        wideRow            := wideRow + 1.U
        when(mode === 6.U && wideRow === 7.U)(state := reading)
          .elsewhen(mode === 7.U && wideRow === 11.U)(state := meanPrepare)
          .otherwise(state := wideRequest)
      }
    }
    when(read.io.req.fire && state === reading)(requested(port) := true.B)
    when(read.io.resp.fire && state === reading) {
      words(port)    := read.io.resp.bits.data
      received(port) := true.B
    }
  }
  val coefficients = blink.bankRead(1)
  when(state === coefficientRequest && coefficients.io.req.fire)(state := coefficientResponse)
  when(state === coefficientResponse && coefficients.io.resp.fire) {
    for (i <- 0 until 4) {
      val index = Mux(stream, i.U(4.W), Cat(coefficientRow(1, 0), i.U(2.W)))
      val value = coefficients.io.resp.bits.data(32 * i + 31, 32 * i)
      when(Mux(stream, coefficientRow === 0.U, coefficientRow < 4.U)) {
        assert(value <= 1073741824.U, "FIXED multiplier exceeds 2^30")
        multipliers(index) := value(30, 0)
      }.otherwise {
        assert(value <= 61.U, "FIXED shift exceeds 61")
        shifts(index) := value(5, 0)
      }
    }
    coefficientRow := coefficientRow + 1.U
    state := Mux(coefficientRow === Mux(stream, 1.U, 7.U), reading, coefficientRequest)
  }
  when(state === reading && received(0) && (!binary || received(1))) {
    state := Mux(mode === 6.U, sumAccumulate, prepare)
  }

  multiply.io.request.valid        := false.B
  multiply.io.request.bits.a       := a.asSInt.pad(64).asUInt
  multiply.io.request.bits.b       := Mux(mode <= 2.U, multipliers(channel).pad(64), b.asSInt.pad(64).asUInt)
  multiply.io.request.bits.signedA := true.B
  multiply.io.request.bits.signedB := mode === 3.U
  multiply.io.request.bits.high    := false.B
  multiply.io.response.ready       := state === scaleWait || state === productWait
  for ((unit, i) <- requantMultipliers.zipWithIndex) {
    val coefficient = Mux(stream, (i + 1).U(4.W), Cat(row(1, 0), (i + 1).U(2.W)))
    unit.io.request.bits   := multiply.io.request.bits
    unit.io.request.bits.a := words(0)(32 * (i + 2) - 1, 32 * (i + 1)).asSInt.pad(64).asUInt
    unit.io.request.bits.b := multipliers(coefficient).pad(64)
    unit.io.request.valid  := state === prepare && mode === 0.U
    unit.io.response.ready := state === scaleWait && mode === 0.U
  }
  when(state === prepare) {
    when(mode === 0.U) {
      multiply.io.request.valid := true.B
      when(multiply.io.request.fire) {
        assert(requantMultipliers.map(_.io.request.ready).reduce(_ && _))
        state := scaleWait
      }
    }.elsewhen(mode <= 2.U && multipliers(channel) === 1.U && shifts(channel) === 0.U) {
      scaled := a
      state  := activate
    }.elsewhen(mode <= 3.U) {
      multiply.io.request.valid            := true.B
      when(multiply.io.request.fire)(state := Mux(mode <= 2.U, scaleWait, productWait))
    }.otherwise {
      for (i <- 0 until 4) {
        val left  = words(0)(32 * i + 31, 32 * i)
        val right = words(1)(32 * i + 31, 32 * i)
        output(i) := Mux(mode === 4.U, Mux(left(31), 0.U, left), clamp(left.asSInt +& right.asSInt))
      }
      state := writing
    }
  }
  when(state === scaleWait && multiply.io.response.fire) {
    when(mode === 0.U) {
      output(0) := roundShift(multiply.io.response.bits, shifts(channel))
      for ((unit, i) <- requantMultipliers.zipWithIndex) {
        val coefficient = Mux(stream, (i + 1).U(4.W), Cat(row(1, 0), (i + 1).U(2.W)))
        assert(unit.io.response.valid)
        output(i + 1) := roundShift(unit.io.response.bits, shifts(coefficient))
      }
      state := writing
    }.otherwise {
      scaled := roundShift(multiply.io.response.bits, shifts(channel))
      state  := activate
    }
  }
  when(state === activate) {
    val q = scaled.asSInt
    when(mode === 0.U) {
      result := scaled
      state  := pack
    }.elsewhen(q === 0.S) {
      result := Mux(mode === 1.U, 0.U, unit >> 1)
      state  := pack
    }.elsewhen(q <= -threshold.asSInt) {
      result := 0.U
      state  := pack
    }.elsewhen(q >= threshold.asSInt) {
      result := Mux(mode === 1.U, scaled, unit)
      state  := pack
    }.elsewhen(mode === 1.U) {
      multiply.io.request.valid            := true.B
      multiply.io.request.bits.a           := q.pad(64).asUInt
      multiply.io.request.bits.b           := (q.pad(64) + threshold.asSInt.pad(64)).asUInt
      multiply.io.request.bits.signedB     := true.B
      when(multiply.io.request.fire)(state := productWait)
    }.otherwise {
      val numerator = (q.pad(64) + threshold.asSInt.pad(64)).asUInt
      quotient       := (numerator >> 1)(27, 0)
      low            := numerator(0)
      divideShift    := 1.U
      divideNegative := false.B
      remainder      := 0.U
      divideStep     := 0.U
      state          := divide
    }
  }
  when(state === productWait && multiply.io.response.fire) {
    val value = multiply.io.response.bits
    when(mode === 3.U) {
      result := roundShift(value, fraction)
      state  := pack
    }.otherwise {
      val magnitude = Mux(value(63), -value, value)
      val shift     = (fraction +& 1.U)(4, 0)
      quotient       := (magnitude >> shift)(27, 0)
      low            := (magnitude & ((1.U(65.W) << shift) - 1.U))(24, 0)
      divideShift    := shift
      divideNegative := value(63)
      remainder      := 0.U
      divideStep     := 0.U
      state          := divide
    }
  }
  when(state === divide) {
    val partial       = Cat(remainder, quotient(27, 21))
    val digit         = (partial / 3.U)(6, 0)
    val nextRemainder = (partial - ((digit << 1) + digit))(1, 0)
    val nextQuotient  = Cat(quotient(20, 0), digit)
    quotient   := nextQuotient
    remainder  := nextRemainder
    divideStep := divideStep + 1.U
    when(divideStep === 3.U) {
      val fullRemainder = (nextRemainder.pad(64) << divideShift) | low.pad(64)
      val half          = 3.U(64.W) << (divideShift - 1.U)
      val increment     = fullRemainder > half || (fullRemainder === half && nextQuotient(0))
      val rounded       = (nextQuotient + increment.asUInt).pad(64)
      result := clamp(Mux(divideNegative, -rounded, rounded).asSInt)
      state  := pack
    }
  }
  when(state === pack) {
    output(lane) := result
    when(lane === 3.U)(state := writing).otherwise {
      lane  := lane + 1.U
      state := prepare
    }
  }

  when(state === sumAccumulate) {
    for (i <- 0 until 4) {
      val index = Cat(row(1, 0), i.U(2.W))
      val value = sums(index) +& words(0)(32 * i + 31, 32 * i).asSInt
      assert(value(64) === value(63), "FIXED sum overflow")
      sums(index) := value(63, 0).asSInt
    }
    requested.foreach(_ := false.B)
    received.foreach(_ := false.B)
    when(row +& 1.U === issue.iter) {
      row   := 0.U
      state := writing
    }.otherwise {
      row   := row + 1.U
      state := reading
    }
  }
  when(state === meanPrepare) {
    val value = sums(meanChannel).asUInt
    meanNegative  := value(63)
    meanQuotient  := Mux(value(63), -value, value)
    meanRemainder := 0.U
    meanStep      := 0.U
    state         := meanDivide
  }
  when(state === meanDivide) {
    val partial   = Cat(meanRemainder, meanQuotient(63))
    val divisor   = divisors(meanChannel)
    val digit     = partial >= divisor
    val remainder = Mux(digit, partial - divisor, partial)(31, 0)
    val quotient  = Cat(meanQuotient(62, 0), digit)
    meanRemainder := remainder
    meanQuotient  := quotient
    meanStep      := meanStep + 1.U
    when(meanStep === 63.U) {
      val twice     = Cat(remainder, 0.U(1.W))
      val increment = twice > divisor || (twice === divisor && quotient(0))
      val rounded   = quotient + increment.asUInt
      meanResults(meanChannel) := clamp(Mux(meanNegative, -rounded, rounded).asSInt)
      meanChannel              := meanChannel + 1.U
      state                    := Mux(meanChannel === 15.U, writing, meanPrepare)
    }
  }

  val write = blink.bankWrite(0)
  write.rob_id                   := issue.rob_id
  write.bank_id                  := issue.wr_bank
  write.group_id                 := issue.wr_group
  write.io.req.valid             := state === writing
  write.io.req.bits.addr         := row
  write.io.req.bits.data         := Cat(output.reverse)
  when(mode === 6.U) {
    write.io.req.bits.data := Cat(sums(Cat(row(2, 0), 1.U(1.W))).asUInt, sums(Cat(row(2, 0), 0.U(1.W))).asUInt)
  }.elsewhen(mode === 7.U) {
    write.io.req.bits.data := Cat((0 until 4).reverse.map(i => meanResults(Cat(row(1, 0), i.U(2.W)))))
  }
  write.io.req.bits.mask         := VecInit(Seq.fill(16)(true.B))
  write.io.resp.ready            := state === acknowledge
  when(write.io.req.fire)(state  := acknowledge)
  when(write.io.resp.fire) {
    assert(write.io.resp.bits.ok, "FIXED bank write failed")
    when(row +& 1.U === Mux(mode === 6.U, 8.U, issue.iter))(state := complete).otherwise {
      row                 := row + 1.U
      lane                := 0.U
      requested.foreach(_ := false.B)
      received.foreach(_  := false.B)
      coefficientRow      := 0.U
      state               := Mux(mode >= 6.U, writing, Mux(stream, coefficientRequest, reading))
    }
  }
  when(blink.cmdResp.fire)(state := idle)
  when(blink.cmdReq.fire) {
    val cmd = blink.cmdReq.bits
    assert(cmd.rs2(63, 10) === 0.U, "FIXED mode or reserved bits")
    assert(
      !cmd.rs2(9) || (cmd.rs2(2, 0) <= 2.U && cmd.iter <= (link.bankEntries / 2).U),
      "FIXED streaming coefficient span"
    )
    assert(cmd.rs2(8, 4) >= 1.U && cmd.rs2(8, 4) <= 24.U, "FIXED fraction must be 1..24")
    assert(!cmd.rs2(3) || cmd.rs2(2, 0) === 3.U, "FIXED only multiply broadcasts")
    assert(
      cmd.op1_bank =/= cmd.op2_bank && cmd.op1_bank =/= cmd.wr_bank &&
        (cmd.op2_bank =/= cmd.wr_bank || cmd.rs2(2, 0) === 6.U),
      "FIXED banks must differ"
    )
    assert(cmd.op1_col === 1.U && cmd.op2_col === 1.U && cmd.wr_col === 1.U, "FIXED requires one group per bank")
    assert(cmd.iter =/= 0.U && cmd.iter(1, 0) === 0.U && cmd.iter <= link.bankEntries.U, "FIXED row span")
    assert(cmd.rs2(2, 0) =/= 7.U || cmd.iter === 4.U, "FIXED mean requires four output rows")
    wideRow             := 0.U
    meanChannel         := 0.U
    issue               := cmd
    row                 := 0.U
    lane                := 0.U
    coefficientRow      := 0.U
    requested.foreach(_ := false.B)
    received.foreach(_  := false.B)
    state               := Mux(cmd.rs2(2, 0) >= 6.U, wideRequest, Mux(cmd.rs2(2, 0) <= 2.U, coefficientRequest, reading))
  }
}

object FixedBall extends BallFactory {

  def instantiate(
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts =
    BallPorts(Instantiate(new FixedBall(link, route)).blink, None)

}
