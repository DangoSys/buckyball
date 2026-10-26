package examples.balls.int8mul

import framework.acc.balldomain.bbus.{BallFactory, BallPorts}
import buckyball.balls.int8mul.{Params => ParametersProto}

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import chisel3.util._
import hardfloat.{recFNFromFN, INToRecFN, RecFNToIN}
import hardfloat.consts.{round_near_even, tininess_afterRounding}
import framework.lib.arith.Fma

import framework.acc.balldomain.ball.blink.{BallStatus, BlinkIO, HasBlink}

@instantiable
class Int8MulBall(
  fp: FrontendParam,
  mp: MemDomainParam,
  bp: BallDomainParam,
  id: Int)
    extends Module
    with HasBlink {
  val ball = bp.balls.find(_.id == id).get

  ball.parameters.unpack(classOf[ParametersProto])

  require(ball.inBW == 2, "Int8MulBall requires inBW=2")
  require(ball.outBW == 1, "Int8MulBall requires outBW=1")
  require(mp.bankWidth == 128, "Int8MulBall requires 128-bit bank rows")
  val bankRowBits = log2Ceil(mp.bankEntries)
  require(bankRowBits <= 32, "Int8MulBall gate_row must fit rs2[63:32]")

  @public
  val blink = IO(new BlinkIO(fp, mp, bp, id))

  val idle :: waitForChannels :: gateRequest :: gateResponse :: inputRequest :: inputResponse :: calculate :: calculateDrain :: writeRequest :: writeResponse :: complete :: Nil =
    Enum(11)
  val state                                                                                                                                                                      = RegInit(idle)
  val robId                                                                                                                                                                      = RegInit(0.U(log2Up(fp.rob_entries).W))
  val gateBank                                                                                                                                                                   = RegInit(0.U(fp.bank_id_len.W))
  val inputBank                                                                                                                                                                  = RegInit(0.U(fp.bank_id_len.W))
  val outputBank                                                                                                                                                                 = RegInit(0.U(fp.bank_id_len.W))
  val groups                                                                                                                                                                     = RegInit(0.U(5.W))
  val group                                                                                                                                                                      = RegInit(0.U(5.W))
  val bases                                                                                                                                                                      = Reg(Vec(3, UInt(mp.groupIdWidth.W)))
  val iter                                                                                                                                                                       = RegInit(0.U(fp.iter_len.W))
  val gateRow                                                                                                                                                                    = RegInit(0.U(bankRowBits.W))
  val inputRow                                                                                                                                                                   = RegInit(0.U(log2Ceil(mp.bankEntries).W))
  val ratio                                                                                                                                                                      = Reg(UInt(32.W))
  val gateWord                                                                                                                                                                   = Reg(UInt(128.W))
  val inputWord                                                                                                                                                                  = Reg(UInt(128.W))
  val lane                                                                                                                                                                       = RegInit(0.U(4.W))
  val outputWord                                                                                                                                                                 = Reg(Vec(16, UInt(8.W)))

  val productReg      = Reg(UInt(16.W))
  val productLaneReg  = Reg(UInt(4.W))
  val productValid    = RegInit(false.B)
  val productFloatReg = Reg(UInt(33.W))
  val floatLaneReg    = Reg(UInt(4.W))
  val floatValid      = RegInit(false.B)
  val scaledReg       = Reg(UInt(33.W))
  val scaledLaneReg   = Reg(UInt(4.W))
  val scaledValid     = RegInit(false.B)

  def positiveFinite(value: UInt): Bool =
    !value(31) && value(30, 23) =/= 255.U && value(30, 0) =/= 0.U

  val gateValue      = (gateWord >> (lane << 3))(7, 0).asSInt
  val inputValue     = (inputWord >> (lane << 3))(7, 0).asSInt
  val integerProduct = (gateValue * inputValue).asUInt
  val productToFloat = Module(new INToRecFN(16, 8, 24))

  val scale = Instantiate(new Fma(3, 8, 24))

  val toInt8 = Module(new RecFNToIN(8, 24, 8))

  productToFloat.io.signedIn       := true.B
  productToFloat.io.in             := productReg
  productToFloat.io.roundingMode   := round_near_even
  productToFloat.io.detectTininess := tininess_afterRounding
  scale.io.validin                 := floatValid
  scale.io.op                      := 0.U
  scale.io.a                       := productFloatReg
  scale.io.b                       := recFNFromFN(8, 24, ratio)
  scale.io.c                       := recFNFromFN(8, 24, 0.U)
  scale.io.roundingMode            := round_near_even
  scale.io.detectTininess          := tininess_afterRounding.asUInt
  toInt8.io.in                     := scaledReg
  toInt8.io.roundingMode           := round_near_even
  toInt8.io.signedOut              := true.B

  blink.cmdReq.ready        := state === idle
  blink.cmdResp.valid       := state === complete
  blink.cmdResp.bits.rob_id := robId
  blink.status.idle         := state === idle
  blink.status.running      := state =/= idle && state =/= complete

  for (port <- 0 until 2) {
    blink.bankRead(port).rob_id           := robId
    blink.bankRead(port).bank_id          := Mux(port.U === 0.U, gateBank, inputBank)
    blink.bankRead(port).group_id         := bases(port) + group
    blink.bankRead(port).io.req.valid     := false.B
    blink.bankRead(port).io.req.bits.addr := Mux(port.U === 0.U, gateRow, inputRow)
    blink.bankRead(port).io.resp.ready    := false.B
  }

  blink.bankWrite(0).rob_id           := robId
  blink.bankWrite(0).bank_id          := outputBank
  blink.bankWrite(0).group_id         := bases(2) + group
  blink.bankWrite(0).io.req.valid     := false.B
  blink.bankWrite(0).io.req.bits.addr := inputRow
  blink.bankWrite(0).io.req.bits.data := Cat(outputWord.reverse)
  blink.bankWrite(0).io.req.bits.mask := VecInit(Seq.fill(16)(true.B))
  blink.bankWrite(0).io.resp.ready    := false.B

  switch(state) {
    is(idle) {
      when(blink.cmdReq.fire) {
        val cmd = blink.cmdReq.bits

        if (bankRowBits < 32)
          assert(
            cmd.rs2(63, 32 + bankRowBits) === 0.U,
            s"Int8MulBall reserves rs2 bits 63:${32 + bankRowBits}"
          )
        assert(positiveFinite(cmd.rs2(31, 0)), "Int8MulBall ratio must be finite and positive")
        assert(
          cmd.rs2(31 + bankRowBits, 32) < mp.bankEntries.U,
          "Int8MulBall gate_row must fit one physical bank"
        )
        assert(
          cmd.op1_col =/= 0.U && cmd.op1_col === cmd.op2_col && cmd.op1_col === cmd.wr_col,
          "Int8MulBall bank groups must match"
        )
        assert(
          cmd.op1_col <= mp.bankNum.U,
          "Int8MulBall bank groups exceed physical banks"
        )
        assert(
          cmd.op1_bank =/= cmd.op2_bank && cmd.op1_bank =/= cmd.wr_bank && cmd.op2_bank =/= cmd.wr_bank,
          "Int8MulBall banks must be distinct"
        )
        assert(
          cmd.iter =/= 0.U && cmd.iter <= mp.bankEntries.U,
          "Int8MulBall iter must fit one physical bank"
        )
        robId      := blink.cmdReq.bits.rob_id
        gateBank   := cmd.op1_bank
        inputBank  := cmd.op2_bank
        outputBank := cmd.wr_bank
        groups     := cmd.op1_col
        group      := 0.U
        bases      := VecInit(cmd.op1_group, cmd.op2_group, cmd.wr_group)
        iter       := cmd.iter
        gateRow    := cmd.rs2(31 + bankRowBits, 32)
        inputRow   := 0.U
        ratio      := cmd.rs2(31, 0)
        lane       := 0.U
        state      := waitForChannels
      }
    }
    is(waitForChannels) {
      state := gateRequest
    }
    is(gateRequest) {
      blink.bankRead(0).io.req.valid            := true.B
      when(blink.bankRead(0).io.req.fire)(state := gateResponse)
    }
    is(gateResponse) {
      blink.bankRead(0).io.resp.ready := true.B
      when(blink.bankRead(0).io.resp.fire) {
        gateWord := blink.bankRead(0).io.resp.bits.data
        state    := inputRequest
      }
    }
    is(inputRequest) {
      blink.bankRead(1).io.req.valid            := true.B
      when(blink.bankRead(1).io.req.fire)(state := inputResponse)
    }
    is(inputResponse) {
      blink.bankRead(1).io.resp.ready := true.B
      when(blink.bankRead(1).io.resp.fire) {
        inputWord := blink.bankRead(1).io.resp.bits.data
        lane      := 0.U
        state     := calculate
      }
    }
    is(calculate) {
      productReg        := integerProduct
      productLaneReg    := lane
      productValid      := true.B
      when(lane === 15.U)(state := calculateDrain)
        .otherwise(lane := lane + 1.U)
    }
    is(calculateDrain) {
      when(scaledValid && scaledLaneReg === 15.U)(state := writeRequest)
    }
    is(writeRequest) {
      blink.bankWrite(0).io.req.valid            := true.B
      when(blink.bankWrite(0).io.req.fire)(state := writeResponse)
    }
    is(writeResponse) {
      blink.bankWrite(0).io.resp.ready := true.B
      when(blink.bankWrite(0).io.resp.fire) {
        when(inputRow +& 1.U === iter) {
          inputRow := 0.U
          when(group +& 1.U === groups)(state := complete)
            .otherwise { group := group + 1.U; state := gateRequest }
        }.otherwise { inputRow := inputRow + 1.U; state := inputRequest }
      }
    }
    is(complete) {
      when(blink.cmdResp.fire)(state := idle)
    }
  }

  when(productValid) {
    productFloatReg := productToFloat.io.out
    floatLaneReg    := productLaneReg
  }
  when(scale.io.validout) {
    scaledReg     := scale.io.out
    scaledLaneReg := Pipe(floatValid, floatLaneReg, 3).bits
  }
  when(scaledValid) {
    outputWord(scaledLaneReg) := toInt8.io.out
  }

  productValid := state === calculate
  floatValid   := productValid
  scaledValid  := scale.io.validout
}

object Int8MulBall extends BallFactory {

  def instantiate(
    fp: FrontendParam,
    mp: MemDomainParam,
    bp: BallDomainParam,
    id: Int
  ): BallPorts =
    BallPorts(Instantiate(new Int8MulBall(fp, mp, bp, id)).blink, None)

}
