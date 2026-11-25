package examples.balls.intsoftmax

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam
import framework.ball.sub_rob.ExpansionParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.ball.blink.{BlinkIO, HasBlink}
import framework.ball.bbus.{BallFactory, BallPorts}
import framework.generic.arith.{DivRem, Multiply}
import buckyball.balls.intsoftmax.{Params => ParametersProto}

@instantiable
class IntSoftmaxBall(
  link:  BlinkParam,
  route: BallRouteParam)
    extends Module
    with HasBlink {
  val ball          = route
  ball.parameters.unpack(classOf[ParametersProto])
  require(ball.inBW == 2 && ball.outBW == 1)
  require(ball.instructions.forall(i => (i.funct7 >> 4) == 4))
  require(link.bankWidth == 128 && link.bankMaskLen == 16 && link.bankEntries == 1024)
  require(link.dataBits == 64)
  @public val blink = IO(new BlinkIO(link, route))

  val (idle :: loadScores :: lookupRequest :: lookupResponse :: divideRequest ::
    divideResponse :: multiplyQuotient :: quotientResponse :: multiplyProduct :: productResponse ::
    writeRequest :: writeResponse :: complete :: Nil) = Enum(13)

  val state         = RegInit(idle)
  val rob           = Reg(UInt(link.robIdBits.W))
  val banks         = Reg(Vec(3, UInt(link.bankIdBits.W)))
  val groups        = Reg(Vec(3, UInt(link.groupIdBits.W)))
  val rows          = Reg(UInt(link.iterBits.W))
  val columns       = Reg(UInt(8.W))
  val stride        = Reg(UInt(6.W))
  val row           = Reg(UInt(link.iterBits.W))
  val column        = Reg(UInt(8.W))
  val inputIssued   = Reg(UInt(6.W))
  val inputReceived = Reg(UInt(6.W))
  val scores        = Reg(Vec(128, SInt(22.W)))
  val maximum       = Reg(SInt(22.W))
  val weights       = Reg(Vec(128, UInt(16.W)))
  val sum           = Reg(UInt(23.W))
  val reciprocal    = Reg(UInt(32.W))
  val numerator     = Reg(UInt(32.W))
  val quotient      = Reg(UInt(32.W))
  val output        = Reg(Vec(4, UInt(32.W)))
  val divider       = Instantiate(new DivRem(64))
  val multiplier    = Instantiate(new Multiply(32))

  // Integer decoding of the F32 bit pattern, clamped to signed Q8.
  def q8(bits: UInt): SInt = {
    val exponent  = bits(30, 23)
    val mantissa  = Cat(1.U(1.W), bits(22, 0))
    val shift     = (142.U - exponent)(4, 0)
    val quotient  = mantissa >> shift
    val mask      = ((1.U(25.W) << shift) - 1.U)(23, 0)
    val remainder = mantissa & mask
    val half      = (1.U(25.W) << ((shift - 1.U)(4, 0)))(24, 0)
    val increment = remainder > half || (remainder === half && quotient(0))
    val magnitude =
      Mux(exponent >= 139.U, (1 << 20).U(25.W), Mux(exponent < 118.U, 0.U(25.W), quotient +& increment.asUInt))
    val signed    = magnitude(20, 0).pad(22).asSInt
    Mux(bits(31), -signed, signed)
  }

  // Every Q15 value fits exactly in the F32 mantissa; no FP unit is needed.
  def f32q15(value: UInt): UInt = {
    val highest  = 15.U(5.W) - PriorityEncoder(Reverse(value(15, 0)))
    val exponent = (highest + 112.U(8.W))(7, 0)
    val mantissa = (value.pad(32) << (23.U - highest))(22, 0)
    Mux(value === 0.U, 0.U(32.W), Cat(0.U(1.W), exponent, mantissa))
  }

  blink.cmdReq.ready        := state === idle
  blink.cmdResp.valid       := state === complete
  blink.cmdResp.bits.rob_id := rob
  blink.status.idle         := state === idle
  blink.status.running      := state =/= idle && state =/= complete

  val input = blink.bankRead(0)
  input.rob_id           := rob
  input.bank_id          := banks(0)
  input.group_id         := groups(0)
  input.io.req.valid     := state === loadScores && inputIssued < stride
  input.io.req.bits.addr := row * stride + inputIssued
  input.io.resp.ready    := state === loadScores

  val difference = (maximum.pad(23) - scores(column(6, 0)).pad(23)).asUInt
  val index      = Mux(difference > 4095.U, 4095.U(12.W), difference(11, 0))
  val table      = blink.bankRead(1)
  table.rob_id           := rob
  table.bank_id          := banks(1)
  table.group_id         := groups(1)
  table.io.req.valid     := state === lookupRequest
  table.io.req.bits.addr := index >> 2
  table.io.resp.ready    := state === lookupResponse

  val write = blink.bankWrite(0)
  write.rob_id           := rob
  write.bank_id          := banks(2)
  write.group_id         := groups(2)
  write.io.req.valid     := state === writeRequest
  write.io.req.bits.addr := row * stride + (column >> 2)
  write.io.req.bits.data := Cat(output.reverse)
  write.io.req.bits.mask := VecInit(Seq.fill(16)(true.B))
  write.io.resp.ready    := state === writeResponse

  divider.io.clear                  := false.B
  divider.io.request.valid          := state === divideRequest
  divider.io.request.bits.a         := (BigInt(1) << 32).U(64.W)
  divider.io.request.bits.b         := sum.pad(64)
  divider.io.request.bits.sew       := 3.U
  divider.io.request.bits.signed    := false.B
  divider.io.request.bits.remainder := false.B
  divider.io.response.ready         := state === divideResponse

  val nextNumerator = (weights(column(6, 0)).pad(32) << 15)(31, 0) + (sum >> 1).pad(32)
  multiplier.io.request.valid        := state === multiplyQuotient || state === multiplyProduct
  multiplier.io.request.bits.a       := Mux(state === multiplyQuotient, nextNumerator, quotient).pad(64)
  multiplier.io.request.bits.b       := Mux(state === multiplyQuotient, reciprocal, sum.pad(32)).pad(64)
  multiplier.io.request.bits.signedA := false.B
  multiplier.io.request.bits.signedB := false.B
  multiplier.io.request.bits.high    := false.B
  multiplier.io.response.ready       := state === quotientResponse || state === productResponse

  when(blink.cmdReq.fire) {
    val cmd    = blink.cmdReq.bits
    val k      = cmd.rs2(7, 0)
    val packed = (k +& 3.U) >> 2
    assert(cmd.rs2(63, 8) === 0.U && k > 0.U && k <= 128.U, "INTSOFTMAX illegal width or encoding")
    assert(cmd.iter > 0.U && cmd.iter * packed <= link.bankEntries.U, "INTSOFTMAX rows exceed bank")
    assert(cmd.op1_col === 1.U && cmd.op2_col === 1.U && cmd.wr_col === 1.U)
    assert(cmd.op1_bank =/= cmd.op2_bank && cmd.op1_bank =/= cmd.wr_bank && cmd.op2_bank =/= cmd.wr_bank)
    rob         := cmd.rob_id
    banks(0)    := cmd.op1_bank; banks(1)   := cmd.op2_bank; banks(2)   := cmd.wr_bank
    groups(0)   := cmd.op1_group; groups(1) := cmd.op2_group; groups(2) := cmd.wr_group
    rows        := cmd.iter; columns        := k; stride                := packed
    row         := 0.U; column              := 0.U
    inputIssued := 0.U; inputReceived       := 0.U
    maximum     := (-(1 << 20)).S(22.W)
    state       := loadScores
  }
  when(state === loadScores) {
    when(input.io.req.fire)(inputIssued := inputIssued + 1.U)
    when(input.io.resp.fire) {
      val values   = (0 until 4).map { lane =>
        val bits    = input.io.resp.bits.data(32 * lane + 31, 32 * lane)
        val valid   = (inputReceived << 2) + lane.U < columns
        val decoded = q8(bits)
        when(valid) {
          assert(bits(30, 23) =/= 255.U || bits(22, 0) === 0.U, "INTSOFTMAX NaN score")
          scores((inputReceived << 2) + lane.U) := decoded
        }
        Mux(valid, decoded, (-(1 << 20)).S(22.W))
      }
      val localMax = values.reduce((a, b) => Mux(a > b, a, b))
      maximum       := Mux(maximum > localMax, maximum, localMax)
      inputReceived := inputReceived + 1.U
      when(inputReceived + 1.U === stride) {
        column := 0.U; sum := 0.U; state := lookupRequest
      }
    }
  }
  when(state === lookupRequest && table.io.req.fire)(state            := lookupResponse)
  when(state === lookupResponse && table.io.resp.fire) {
    val entry    = (table.io.resp.bits.data >> (index(1, 0) << 5))(31, 0)
    assert(entry <= 32768.U, "INTSOFTMAX illegal Q15 table entry")
    when(index === 0.U)(assert(entry === 32768.U))
    val selected = entry(15, 0)
    weights(column(6, 0)) := selected
    sum                   := sum + selected
    when(column + 1.U === columns)(state := divideRequest)
      .otherwise { column := column + 1.U; state := lookupRequest }
  }
  when(state === divideRequest && divider.io.request.fire) {
    assert(sum >= 32768.U && sum <= (128 * 32768).U)
    state := divideResponse
  }
  when(state === divideResponse && divider.io.response.fire) {
    reciprocal       := divider.io.response.bits(31, 0)
    column           := 0.U
    output.foreach(_ := 0.U)
    state            := multiplyQuotient
  }
  when(state === multiplyQuotient && multiplier.io.request.fire) {
    numerator := nextNumerator
    state     := quotientResponse
  }
  when(state === quotientResponse && multiplier.io.response.fire) {
    quotient := multiplier.io.response.bits(63, 32)
    state    := multiplyProduct
  }
  when(state === multiplyProduct && multiplier.io.request.fire)(state := productResponse)
  when(state === productResponse && multiplier.io.response.fire) {
    val product   = multiplier.io.response.bits
    assert(numerator.pad(64) >= product)
    val corrected = quotient +& (numerator.pad(64) - product >= sum).asUInt
    assert(corrected <= 32768.U)
    output(column(1, 0)) := f32q15(corrected(15, 0))
    when(column(1, 0) === 3.U || column + 1.U === columns)(state := writeRequest)
      .otherwise { column := column + 1.U; state := multiplyQuotient }
  }
  when(state === writeRequest && write.io.req.fire)(state             := writeResponse)
  when(state === writeResponse && write.io.resp.fire) {
    assert(write.io.resp.bits.ok)
    when(column + 1.U === columns) {
      when(row + 1.U === rows)(state := complete)
        .otherwise {
          row         := row + 1.U
          inputIssued := 0.U; inputReceived := 0.U
          maximum     := (-(1 << 20)).S(22.W)
          state       := loadScores
        }
    }.otherwise {
      column           := column + 1.U
      output.foreach(_ := 0.U)
      state            := multiplyQuotient
    }
  }
  when(state === complete && blink.cmdResp.fire)(state                := idle)
}

object IntSoftmaxBall extends BallFactory {

  def instantiate(
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts =
    BallPorts(Instantiate(new IntSoftmaxBall(link, route)).blink, None)

}
