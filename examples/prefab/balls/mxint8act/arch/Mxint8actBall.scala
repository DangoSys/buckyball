package examples.balls.mxint8act

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.ball.bbus.{BallFactory, BallPorts}
import framework.ball.blink.{BlinkIO, HasBlink}
import framework.ball.sub_rob.ExpansionParam

/** Packed MXINT8 activation and gate; all arithmetic is integer arithmetic. */
@instantiable
class Mxint8actBall(
  link:  BlinkParam,
  route: BallRouteParam)
    extends Module
    with HasBlink {
  require(link.bankWidth == 128 && link.bankMaskLen == 16)
  require(link.bankEntries >= 1024, "Mxint8actBall keeps its 4096-entry table in one bank of at least 1024 rows")
  @public val blink = IO(new BlinkIO(link, route))

  val (idle :: inputReq :: inputWait :: startLane :: tableReq :: tableWait ::
    tableNextReq :: tableNextWait :: multiply :: interpolate :: rescale :: commit ::
    selectScale :: encode :: writeReq :: writeWait :: complete :: Nil) = Enum(17)

  val state          = RegInit(idle)
  val rob            = Reg(UInt(link.robIdBits.W))
  val banks          = Reg(Vec(3, UInt(link.bankIdBits.W)))
  val groups         = Reg(Vec(3, UInt(link.groupIdBits.W)))
  val inputBase      = Reg(UInt(32.W))
  val outputBase     = Reg(UInt(32.W))
  val tableBase      = Reg(UInt(32.W))
  val k              = Reg(UInt(16.W))
  val inputStride    = Reg(UInt(18.W))
  val outputStride   = Reg(UInt(18.W))
  val rows           = Reg(UInt(link.iterBits.W))
  val row            = Reg(UInt(link.iterBits.W))
  val block          = Reg(UInt(16.W))
  val kind           = Reg(UInt(3.W))
  val lane           = Reg(UInt(5.W))
  val gated          = Reg(Bool())
  val codes          = Reg(Vec(32, SInt(8.W)))
  val gates          = Reg(Vec(32, SInt(8.W)))
  val values         = Reg(Vec(32, SInt(64.W)))
  val output         = Reg(Vec(32, UInt(8.W)))
  val scales         = Reg(Vec(16, UInt(8.W)))
  val exponent       = Reg(SInt(9.W))
  val gateExponent   = Reg(SInt(9.W))
  val outputExponent = Reg(SInt(9.W))
  val index          = Reg(UInt(12.W))
  val fraction       = Reg(UInt(19.W))
  val first          = Reg(SInt(32.W))
  val last           = Reg(SInt(32.W))
  val product        = Reg(SInt(64.W))
  val interpolated   = Reg(SInt(64.W))
  val activated      = Reg(SInt(64.W))
  val maximum        = Reg(UInt(64.W))
  val writeKind      = Reg(UInt(2.W))
  val blocks         = k >> 5
  val scale          = exponent.pad(11) + Mux(gated, gateExponent.pad(11), 0.S(11.W))

  def roundRight(value: SInt, amount: UInt): SInt = {
    val magnitude = Mux(value < 0.S, -value, value).asUInt
    val shift     = amount.pad(6)(5, 0)
    val quotient  = magnitude >> shift
    val mask      = ((1.U(64.W) << shift)(63, 0)) - 1.U
    val remainder = magnitude & mask
    val half      = (1.U(64.W) << (shift - 1.U)(5, 0))(63, 0)
    val increment = amount =/= 0.U && (remainder > half || (remainder === half && quotient(0)))
    val rounded   = (quotient + increment.asUInt)(63, 0).asSInt
    Mux(amount >= 64.U, 0.S(64.W), Mux(value < 0.S, -rounded, rounded))
  }

  blink.cmdReq.ready        := state === idle
  blink.cmdResp.valid       := state === complete
  blink.cmdResp.bits.rob_id := rob
  blink.status.idle         := state === idle
  blink.status.running      := state =/= idle
  for (port <- 0 until 2) {
    val read = blink.bankRead(port)
    read.rob_id           := rob
    read.bank_id          := banks(port)
    read.group_id         := groups(port)
    read.io.req.valid     := false.B
    read.io.req.bits.addr := 0.U
    read.io.resp.ready    := false.B
  }
  val inputRead = blink.bankRead(0)
  val source = inputBase + row * inputStride

  val sourceOffset = MuxLookup(kind, 0.U)(Seq(
    0.U -> (block << 1),
    1.U -> ((block << 1) + 1.U),
    2.U -> ((k >> 4) + (block >> 4)),
    3.U -> (rows * inputStride + (block << 1)),
    4.U -> (rows * inputStride + (block << 1) + 1.U),
    5.U -> (rows * inputStride + (k >> 4) + (block >> 4))
  ))

  inputRead.io.req.valid     := state === inputReq
  inputRead.io.req.bits.addr := source + sourceOffset
  inputRead.io.resp.ready    := state === inputWait
  val tableRead = blink.bankRead(1)
  tableRead.io.req.valid     := state === tableReq || state === tableNextReq
  tableRead.io.req.bits.addr := tableBase + (index >> 2) + (state === tableNextReq).asUInt
  tableRead.io.resp.ready    := state === tableWait || state === tableNextWait
  val write = blink.bankWrite(0)
  write.bank_id          := banks(2)
  write.group_id         := groups(2)
  write.rob_id           := rob
  write.io.req.valid     := state === writeReq
  write.io.req.bits.addr := outputBase + row * outputStride +
    Mux(writeKind === 2.U, (k >> 4) + (block >> 4), (block << 1) + writeKind)
  write.io.req.bits.data := Mux(
    writeKind === 2.U,
    Cat(scales.reverse),
    Mux(writeKind === 0.U, Cat(output.take(16).reverse), Cat(output.drop(16).reverse))
  )
  write.io.req.bits.mask := VecInit(Seq.fill(16)(true.B))
  write.io.resp.ready    := state === writeWait

  switch(state) {
    is(idle) {
      when(blink.cmdReq.fire) {
        val cmd     = blink.cmdReq.bits
        val width   = cmd.rs2(16, 1)
        val inRows  = (width >> 4) +& ((width +& 511.U) >> 9)
        val outRows = (width >> 4) +& ((width +& 511.U) >> 9)
        assert(cmd.rs2(63, 59) === 0.U, "MXINT8ACT reserved encoding")
        assert(width > 0.U && width(4, 0) === 0.U && cmd.iter > 0.U, "MXINT8ACT dimensions")
        assert(cmd.op1_col === 1.U && cmd.op2_col === 1.U && cmd.wr_col === 1.U)
        assert(cmd.op1_bank =/= cmd.op2_bank && cmd.op1_bank =/= cmd.wr_bank && cmd.op2_bank =/= cmd.wr_bank)
        assert(cmd.rs2(30, 17) +& (cmd.iter * inRows << cmd.rs2(0)) <= link.bankEntries.U, "MXINT8ACT input span")
        assert(cmd.rs2(44, 31) +& cmd.iter * outRows <= link.bankEntries.U, "MXINT8ACT output span")
        assert(cmd.rs2(58, 45) +& 1024.U <= link.bankEntries.U, "MXINT8ACT table span")
        rob          := cmd.rob_id
        banks        := VecInit(cmd.op1_bank, cmd.op2_bank, cmd.wr_bank)
        groups       := VecInit(cmd.op1_group, cmd.op2_group, cmd.wr_group)
        inputBase    := cmd.rs2(30, 17)
        outputBase   := cmd.rs2(44, 31)
        tableBase    := cmd.rs2(58, 45)
        k            := width
        inputStride  := inRows
        outputStride := outRows
        rows         := cmd.iter
        row          := 0.U; block := 0.U; kind := 0.U
        gated        := cmd.rs2(0)
        scales       := VecInit(Seq.fill(16)(0.U(8.W)))
        state        := inputReq
      }
    }
    is(inputReq)(when(inputRead.io.req.fire)(state := inputWait))
    is(inputWait) {
      when(inputRead.io.resp.fire) {
        val data = inputRead.io.resp.bits.data
        when(kind === 0.U || kind === 1.U) {
          for (i <- 0 until 16) codes(Cat(kind(0), i.U(4.W))) := data(i * 8 + 7, i * 8).asSInt
          kind := kind + 1.U; state := inputReq
        }.elsewhen(kind === 2.U) {
          val encoded = (data >> (block(3, 0) << 3))(7, 0)
          assert(encoded =/= 255.U, "MXINT8ACT exponent must be finite")
          exponent := encoded.zext - 127.S
          when(gated) { kind := 3.U; state := inputReq }.otherwise {
            lane := 0.U; maximum := 0.U; state := startLane
          }
        }.elsewhen(kind === 3.U || kind === 4.U) {
          for (i <- 0 until 16) gates(Cat((kind === 4.U), i.U(4.W))) := data(i * 8 + 7, i * 8).asSInt
          kind := kind + 1.U; state := inputReq
        }.otherwise {
          val encoded = (data >> (block(3, 0) << 3))(7, 0)
          assert(encoded =/= 255.U, "MXINT8ACT gate exponent must be finite")
          gateExponent := encoded.zext - 127.S
          lane         := 0.U; maximum := 0.U; state := startLane
        }
      }
    }
    is(startLane) {
      val code  = codes(lane).pad(64)
      assert(code =/= -128.S, "MXINT8ACT code -128 is reserved")
      val shift = (Mux(exponent > 4.S, 4.S, exponent) + 26.S).asUInt
      val x     = (code << shift(4, 0))(63, 0).asSInt
      when(exponent < -24.S) { activated := (code << 15)(63, 0).asSInt; state := commit }
        .elsewhen(x >= (16L << 26).S) { activated := (code << 16)(63, 0).asSInt; state := commit }
        .elsewhen(x <= -(16L << 26).S) { activated := 0.S; state := commit }
        .otherwise {
          val position = (x + (16L << 26).S).asUInt
          index := position(30, 19); fraction := position(18, 0); state := tableReq
        }
    }
    is(tableReq)(when(tableRead.io.req.fire)(state := tableWait))
    is(tableWait) {
      when(tableRead.io.resp.fire) {
        val words = tableRead.io.resp.bits.data.asTypeOf(Vec(4, SInt(32.W)))
        first              := words(index(1, 0))
        when(index === 4095.U) { last := (16L << 26).S; state := multiply }
          .elsewhen(index(1, 0) =/= 3.U) { last := words((index(1, 0) + 1.U)(1, 0)); state := multiply }
          .otherwise(state := tableNextReq)
      }
    }
    is(tableNextReq)(when(tableRead.io.req.fire)(state := tableNextWait))
    is(tableNextWait) {
      when(tableRead.io.resp.fire) { last := tableRead.io.resp.bits.data(31, 0).asSInt; state := multiply }
    }
    is(multiply) { product := (last.pad(33) - first.pad(33)) * fraction.zext; state := interpolate }
    is(interpolate) { interpolated := first.pad(64) + roundRight(product, 19.U); state := rescale }
    is(rescale) {
      val shift = exponent.pad(11) + 10.S
      activated := Mux(
        shift >= 0.S,
        roundRight(interpolated, shift.asUInt),
        (interpolated << (-shift).asUInt(3, 0))(63, 0).asSInt
      )
      state     := commit
    }
    is(commit) {
      when(gated)(assert(gates(lane) =/= -128.S, "MXINT8ACT gate code -128 is reserved"))
      val value = Mux(gated, (activated * gates(lane))(63, 0).asSInt, activated)
      values(lane) := value
      val magnitude = Mux(value < 0.S, -value, value).asUInt
      when(magnitude > maximum)(maximum := magnitude)
      when(lane === 31.U)(state := selectScale).otherwise { lane := lane + 1.U; state := startLane }
    }
    is(selectScale) {
      val highest   = 63.U(6.W) - PriorityEncoder(Reverse(maximum))
      val candidate = scale + highest.zext - 22.S
      outputExponent := Mux(
        maximum === 0.U,
        0.S,
        Mux(candidate < -127.S, -127.S, Mux(candidate > 127.S, 127.S, candidate))
      )
      lane           := 0.U; state := encode
    }
    is(encode) {
      val shift     = outputExponent.pad(11) - scale + 16.S
      val value     = values(lane)
      val rounded   = Mux(shift >= 0.S, roundRight(value, shift.asUInt), (value << (-shift).asUInt(2, 0))(63, 0).asSInt)
      val saturated = Mux(value === 0.S, 0.S, Mux(value < 0.S, -127.S, 127.S))
      val code      = Mux(shift < -6.S, saturated, Mux(rounded > 127.S, 127.S, Mux(rounded < -127.S, -127.S, rounded)))
      output(lane)     := code.asUInt(7, 0)
      when(lane === 31.U) {
        scales(block(3, 0)) := (outputExponent + 127.S).asUInt
        writeKind           := 0.U; state := writeReq
      }.otherwise(lane := lane + 1.U)
    }
    is(writeReq)(when(write.io.req.fire)(state := writeWait))
    is(writeWait) {
      when(write.io.resp.fire) {
        when(writeKind === 0.U) { writeKind := 1.U; state := writeReq }
          .elsewhen(writeKind === 1.U && (block(3, 0) === 15.U || block + 1.U === blocks)) {
            writeKind := 2.U; state := writeReq
          }.otherwise {
            when(writeKind === 2.U)(scales := VecInit(Seq.fill(16)(0.U(8.W))))
            when(block + 1.U === blocks) {
              when(row + 1.U === rows)(state := complete).otherwise {
                row := row + 1.U; block := 0.U; kind := 0.U; state := inputReq
              }
            }.otherwise { block := block + 1.U; kind := 0.U; state := inputReq }
          }
      }
    }
    is(complete)(when(blink.cmdResp.fire)(state := idle))
  }
}

object Mxint8actBall extends BallFactory {

  def instantiate(
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts =
    BallPorts(Instantiate(new Mxint8actBall(link, route)).blink, None)

}
