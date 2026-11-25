package examples.balls.lut

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam

import framework.ball.bbus.{BallFactory, BallPorts}
import buckyball.balls.lut.{Params => ParametersProto}
import framework.ball.blink.{BlinkIO, HasBlink}
import framework.ball.sub_rob.ExpansionParam
import framework.generic.arith.{DivRem, Multiply}
import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import chisel3.util._

@instantiable
class LutBall(
  link:  BlinkParam,
  route: BallRouteParam)
    extends Module
    with HasBlink {
  val ball = route
  ball.parameters.unpack(classOf[ParametersProto])
  require(link.bankWidth == 128, "LutBall requires 128-bit bank rows")
  require(link.bankEntries >= 16, "LutBall requires sixteen LUT rows")
  require(ball.inBW == 2 && ball.outBW == 1, "LutBall requires two reads and one write")

  @public val blink                                                                                                                                            = IO(new BlinkIO(link, route))
  val idle :: inputReq :: inputResp :: lutReq :: lutResp :: writeReq :: writeResp :: divideReq :: divideResp :: multiplyReq :: multiplyResp :: complete :: Nil =
    Enum(12)
  val state                                                                                                                                                    = RegInit(idle)
  val robId                                                                                                                                                    = Reg(UInt(link.robIdBits.W))
  val inputBank                                                                                                                                                = Reg(UInt(link.bankIdBits.W))
  val lutBank                                                                                                                                                  = Reg(UInt(link.bankIdBits.W))
  val outputBank                                                                                                                                               = Reg(UInt(link.bankIdBits.W))
  val bases                                                                                                                                                    = Reg(Vec(3, UInt(link.groupIdBits.W)))
  val iter                                                                                                                                                     = Reg(UInt(link.iterBits.W))
  val inputRow                                                                                                                                                 = Reg(UInt(log2Ceil(link.bankEntries).W))
  val inputData                                                                                                                                                = Reg(UInt(128.W))
  val laneTable                                                                                                                                                = Reg(Bool())
  val lookupLane                                                                                                                                               = Reg(UInt(4.W))
  val outputWord                                                                                                                                               = Reg(Vec(16, UInt(8.W)))
  val mode                                                                                                                                                     = Reg(UInt(2.W))
  // Softmax phases: maximum, table lookup, normalization, scratch cleanup.
  val phase                                                                                                                                                    = Reg(UInt(2.W))
  val maximum                                                                                                                                                  = Reg(UInt(32.W))
  val sum                                                                                                                                                      = Reg(UInt(64.W))
  val multiplier                                                                                                                                               = Reg(UInt(30.W))
  val reciprocal                                                                                                                                               = Reg(UInt(32.W))
  val divide                                                                                                                                                   = Instantiate(new DivRem(64))
  val multiply                                                                                                                                                 = Instantiate(new Multiply(32))
  val word                                                                                                                                                     = (inputData >> (lookupLane(1, 0) << 5))(31, 0)
  val bounded                                                                                                                                                  = Mux(word.asSInt < -512.S, -512.S(32.W), Mux(word.asSInt > 511.S, 511.S(32.W), word.asSInt)).asUInt
  val distance                                                                                                                                                 = (maximum.asSInt.pad(33) - word.asSInt.pad(33)).asUInt
  val relative                                                                                                                                                 = Mux(distance > 1023.U, 1023.U, distance)(9, 0)
  val tableIndex                                                                                                                                               = Mux(mode === 2.U, relative, bounded(9, 0))
  val inputByte                                                                                                                                                = (inputData >> (lookupLane << 3))(7, 0)
  val normalize                                                                                                                                                = mode === 2.U && phase === 2.U
  val cleanup                                                                                                                                                  = mode === 2.U && phase === 3.U

  blink.cmdReq.ready        := state === idle
  blink.cmdResp.valid       := state === complete
  blink.cmdResp.bits.rob_id := robId
  blink.status.idle         := state === idle
  blink.status.running      := state =/= idle && state =/= complete
  for (port <- 0 until 2) {
    blink.bankRead(port).rob_id           := robId
    blink.bankRead(port).group_id         := bases(port)
    blink.bankRead(port).io.req.valid     := false.B
    blink.bankRead(port).io.req.bits.addr := 0.U
    blink.bankRead(port).io.resp.ready    := false.B
  }
  blink.bankRead(0).bank_id := Mux(normalize, outputBank, inputBank)
  blink.bankRead(0).group_id := Mux(normalize, bases(2), bases(0))
  blink.bankRead(1).bank_id  := lutBank
  blink.bankRead(1).group_id := bases(1) + Mux(mode === 0.U && laneTable, lookupLane >> 2, 0.U)
  val write = blink.bankWrite(0)
  write.rob_id                     := robId
  write.bank_id                    := outputBank
  write.group_id                   := bases(2)
  write.io.req.valid               := state === writeReq
  write.io.req.bits.addr           := Mux(normalize, inputRow >> 2, inputRow)
  write.io.req.bits.data           := Mux(cleanup, 0.U, Cat(outputWord.reverse))
  write.io.req.bits.mask           := VecInit(Seq.fill(link.bankMaskLen)(true.B))
  write.io.resp.ready              := state === writeResp
  divide.io.clear                  := false.B
  divide.io.request.valid          := state === divideReq
  divide.io.request.bits.a         := (multiplier << 32).pad(64)
  divide.io.request.bits.b         := sum
  divide.io.request.bits.sew       := 3.U
  divide.io.request.bits.signed    := false.B
  divide.io.request.bits.remainder := false.B
  divide.io.response.ready         := state === divideResp
  multiply.io.request.valid        := state === multiplyReq
  multiply.io.request.bits.a       := word.pad(64)
  multiply.io.request.bits.b       := reciprocal.pad(64)
  multiply.io.request.bits.signedA := false.B
  multiply.io.request.bits.signedB := false.B
  multiply.io.request.bits.high    := false.B
  multiply.io.response.ready       := state === multiplyResp

  switch(state) {
    is(idle) {
      when(blink.cmdReq.fire) {
        val cmd      = blink.cmdReq.bits
        val selected = cmd.rs2(1, 0)
        assert(selected <= 2.U && cmd.rs2(31, 2) === 0.U, "LUT reserved encoding")
        assert(cmd.iter > 0.U && cmd.iter <= link.bankEntries.U, "LUT rows exceed bank depth")
        assert(cmd.op1_col === 1.U && cmd.wr_col === 1.U)
        assert(cmd.op1_bank =/= cmd.op2_bank && cmd.op1_bank =/= cmd.wr_bank && cmd.op2_bank =/= cmd.wr_bank)
        when(selected === 0.U) {
          assert(cmd.op2_col === 1.U || cmd.op2_col === 4.U)
        }.otherwise {
          assert(link.bankEntries.U >= 256.U && cmd.op2_col === 1.U)
        }
        when(selected === 2.U) {
          assert(cmd.rs2(63, 32) > 0.U && cmd.rs2(63, 32) < (1.U << 30))
        }.otherwise {
          assert(cmd.rs2(63, 32) === 0.U)
        }
        robId      := cmd.rob_id
        inputBank  := cmd.op1_bank
        lutBank    := cmd.op2_bank
        outputBank := cmd.wr_bank
        bases      := VecInit(cmd.op1_group, cmd.op2_group, cmd.wr_group)
        iter       := cmd.iter
        inputRow   := 0.U
        laneTable  := cmd.op2_col === 4.U
        lookupLane := 0.U
        outputWord := VecInit(Seq.fill(16)(0.U(8.W)))
        mode       := selected
        phase      := Mux(selected === 2.U, 0.U, 1.U)
        maximum    := "h80000000".U
        sum        := 0.U
        multiplier := cmd.rs2(61, 32)
        state      := inputReq
      }
    }
    is(inputReq) {
      blink.bankRead(0).io.req.valid            := true.B
      blink.bankRead(0).io.req.bits.addr        := inputRow
      when(blink.bankRead(0).io.req.fire)(state := inputResp)
    }
    is(inputResp) {
      blink.bankRead(0).io.resp.ready := true.B
      when(blink.bankRead(0).io.resp.fire) {
        val data = blink.bankRead(0).io.resp.bits.data
        inputData  := data
        lookupLane := 0.U
        when(mode === 2.U && phase === 0.U) {
          val peak = (0 until 4).map(i => data(32 * i + 31, 32 * i).asSInt)
            .reduce((a, b) => Mux(a > b, a, b))
          maximum              := Mux(peak > maximum.asSInt, peak.asUInt, maximum)
          when(inputRow +& 1.U === iter) {
            inputRow := 0.U
            phase    := 1.U
          }.otherwise(inputRow := inputRow + 1.U)
          state                := inputReq
        }.otherwise {
          state := Mux(normalize, multiplyReq, lutReq)
        }
      }
    }
    is(lutReq) {
      blink.bankRead(1).io.req.valid            := true.B
      blink.bankRead(1).io.req.bits.addr        := Mux(
        mode === 0.U,
        Mux(laneTable, Cat(lookupLane(1, 0), inputByte(7, 4)), inputByte(7, 4)),
        tableIndex(9, 2)
      )
      when(blink.bankRead(1).io.req.fire)(state := lutResp)
    }
    is(lutResp) {
      blink.bankRead(1).io.resp.ready := true.B
      when(blink.bankRead(1).io.resp.fire) {
        val data = blink.bankRead(1).io.resp.bits.data
        when(mode === 0.U) {
          outputWord(lookupLane) := (data >> (inputByte(3, 0) << 3))(7, 0)
        }.otherwise {
          val value = (data >> (tableIndex(1, 0) << 5))(31, 0)
          for (byte <- 0 until 4) {
            outputWord(Cat(lookupLane(1, 0), byte.U(2.W))) := value(8 * byte + 7, 8 * byte)
          }
          when(mode === 2.U) {
            assert(value <= (1.U << 30), "LUT normalized entry exceeds Q30")
            when(tableIndex === 0.U)(assert(value === (1.U << 30), "LUT normalized entry zero must be Q30 one"))
            sum := sum + value
          }
        }
        when(lookupLane === Mux(mode === 0.U, 15.U, 3.U))(state := writeReq)
          .otherwise { lookupLane := lookupLane + 1.U; state := lutReq }
      }
    }
    is(divideReq) {
      when(divide.io.request.fire)(state := divideResp)
    }
    is(divideResp) {
      when(divide.io.response.fire) {
        assert(divide.io.response.bits(63, 32) === 0.U)
        reciprocal := divide.io.response.bits(31, 0)
        inputRow   := 0.U
        outputWord := VecInit(Seq.fill(16)(0.U(8.W)))
        phase      := 2.U
        state      := inputReq
      }
    }
    is(multiplyReq) {
      when(multiply.io.request.fire)(state := multiplyResp)
    }
    is(multiplyResp) {
      when(multiply.io.response.fire) {
        val product   = multiply.io.response.bits
        val quotient  = product(63, 48)
        val remainder = product(47, 0)
        val rounded   = quotient +& (remainder > (1.U << 47) || (remainder === (1.U << 47) && quotient(0)))
        outputWord(Cat(inputRow(1, 0), lookupLane(1, 0))) := Mux(rounded > 127.U, 127.U, rounded(7, 0))
        when(lookupLane === 3.U) {
          when(inputRow(1, 0) === 3.U || inputRow +& 1.U === iter)(state := writeReq)
            .otherwise { inputRow := inputRow + 1.U; state := inputReq }
        }.otherwise { lookupLane := lookupLane + 1.U; state := multiplyReq }
      }
    }
    is(writeReq) {
      when(write.io.req.fire)(state := writeResp)
    }
    is(writeResp) {
      when(write.io.resp.fire) {
        when(mode === 2.U && phase === 1.U && inputRow +& 1.U === iter) {
          state := divideReq
        }.elsewhen(normalize) {
          when(inputRow +& 1.U === iter) {
            val firstUnused = (iter +& 3.U) >> 2
            when(firstUnused < iter) {
              inputRow := firstUnused
              phase    := 3.U
              state    := writeReq
            }.otherwise(state := complete)
          }.otherwise {
            inputRow   := inputRow + 1.U
            outputWord := VecInit(Seq.fill(16)(0.U(8.W)))
            state      := inputReq
          }
        }.elsewhen(inputRow +& 1.U === iter) {
          state := complete
        }.otherwise {
          inputRow := inputRow + 1.U
          state    := Mux(cleanup, writeReq, inputReq)
        }
      }
    }
    is(complete) {
      when(blink.cmdResp.fire)(state := idle)
    }
  }
}

object LutBall extends BallFactory {

  def instantiate(
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts =
    BallPorts(Instantiate(new LutBall(link, route)).blink, None)

}
