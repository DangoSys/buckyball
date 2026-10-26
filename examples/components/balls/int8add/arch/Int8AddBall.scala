package examples.balls.int8add

import framework.acc.balldomain.bbus.{BallFactory, BallPorts}
import buckyball.balls.int8add.{Params => ParametersProto}

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import chisel3.util._
import hardfloat.{rawFloatFromRecFN, recFNFromFN, RecFNToIN, RoundRawFNToRecFN}
import hardfloat.consts.{round_near_even, tininess_afterRounding}
import framework.lib.arith.{Fma, IntToFloat}

import framework.acc.balldomain.ball.blink.{BallStatus, BlinkIO, HasBlink}

@instantiable
class Int8AddBall(
  fp: FrontendParam,
  mp: MemDomainParam,
  bp: BallDomainParam,
  id: Int)
    extends Module
    with HasBlink {
  val ball = bp.balls.find(_.id == id).get

  ball.parameters.unpack(classOf[ParametersProto])

  val addFunct = ball.instructions
    .find(_.mnemonic == "INT8ADD").get.funct7

  val reluFunct = ball.instructions
    .find(_.mnemonic == "INT8ADD_RELU").get.funct7

  require(ball.inBW == 2, "Int8AddBall requires inBW=2")
  require(ball.outBW == 1, "Int8AddBall requires outBW=1")
  require(mp.bankWidth == 128, "Int8AddBall requires 128-bit bank rows")
  require((addFunct >> 4) == 4 && (reluFunct >> 4) == 4, "Int8AddBall instructions must encode two reads and one write")

  @public
  val blink = IO(new BlinkIO(fp, mp, bp, id))

  val idle :: waitForChannels :: readRequest :: readResponse :: calculate :: calculateDrain :: writeRequest :: writeResponse :: complete :: Nil =
    Enum(9)
  val state                                                                                                                                     = RegInit(idle)
  val robId                                                                                                                                     = RegInit(0.U(log2Up(fp.rob_entries).W))
  val lhsBank                                                                                                                                   = RegInit(0.U(fp.bank_id_len.W))
  val rhsBank                                                                                                                                   = RegInit(0.U(fp.bank_id_len.W))
  val outputBank                                                                                                                                = RegInit(0.U(fp.bank_id_len.W))
  val groups                                                                                                                                    = RegInit(0.U(5.W))
  val group                                                                                                                                     = RegInit(0.U(5.W))
  val bases                                                                                                                                     = Reg(Vec(3, UInt(mp.groupIdWidth.W)))
  val iter                                                                                                                                      = RegInit(0.U(fp.iter_len.W))
  val row                                                                                                                                       = RegInit(0.U(log2Ceil(mp.bankEntries).W))
  val lhsRatio                                                                                                                                  = Reg(UInt(32.W))
  val rhsRatio                                                                                                                                  = Reg(UInt(32.W))
  val relu                                                                                                                                      = RegInit(false.B)
  val lhsWord                                                                                                                                   = Reg(UInt(128.W))
  val rhsWord                                                                                                                                   = Reg(UInt(128.W))
  val lhsRequested                                                                                                                              = RegInit(false.B)
  val rhsRequested                                                                                                                              = RegInit(false.B)
  val lhsReceived                                                                                                                               = RegInit(false.B)
  val rhsReceived                                                                                                                               = RegInit(false.B)
  val lane                                                                                                                                      = RegInit(0.U(4.W))
  val completed                                                                                                                                 = RegInit(0.U(5.W))
  val outputWord                                                                                                                                = Reg(Vec(16, UInt(8.W)))

  def positiveFinite(value: UInt): Bool =
    !value(31) && value(30, 23) =/= 255.U && value(30, 0) =/= 0.U

  val lhsToFloat = Instantiate(new IntToFloat(8, 8, 24))

  val rhsToFloat = Instantiate(new IntToFloat(8, 8, 24))

  val lhsMul = Instantiate(new Fma(3, 8, 24))

  val rhsMul = Instantiate(new Fma(3, 8, 24))

  val add = Instantiate(new PipelinedAddRawFN(8, 24))

  val round          = Instantiate(new RoundRawFNToRecFN(8, 24, 0))
  val toInt          = Instantiate(new RecFNToIN(8, 24, 8))
  val lhsValue       = (lhsWord >> (lane << 3))(7, 0)
  val rhsValue       = (rhsWord >> (lane << 3))(7, 0)
  val calculateIssue = state === calculate

  val ratioStage0    = RegEnable(lhsRatio, calculateIssue)
  val ratioStage1    = RegEnable(ratioStage0, RegNext(calculateIssue, false.B))
  val rhsRatioStage0 = RegEnable(rhsRatio, calculateIssue)
  val rhsRatioStage1 = RegEnable(rhsRatioStage0, RegNext(calculateIssue, false.B))
  val laneStage0     = RegEnable(lane, calculateIssue)
  val laneStage1     = RegEnable(laneStage0, RegNext(calculateIssue, false.B))

  val conversionValid = lhsToFloat.io.validout
  val productLane     = Pipe(conversionValid, laneStage1, 3)

  val addResult     = add.io.rawOut
  val addLane       = Pipe(lhsMul.io.validout, productLane.bits, 4)
  val addValid      = add.io.validout
  val roundedResult = RegEnable(round.io.out, addValid)
  val roundedLane   = RegEnable(addLane, addValid)
  val resultValid   = RegNext(addValid, false.B)

  lhsToFloat.io.validin        := calculateIssue
  lhsToFloat.io.signedIn       := true.B
  lhsToFloat.io.in             := lhsValue
  lhsToFloat.io.roundingMode   := round_near_even
  lhsToFloat.io.detectTininess := tininess_afterRounding
  rhsToFloat.io.validin        := calculateIssue
  rhsToFloat.io.signedIn       := true.B
  rhsToFloat.io.in             := rhsValue
  rhsToFloat.io.roundingMode   := round_near_even
  rhsToFloat.io.detectTininess := tininess_afterRounding

  lhsMul.io.validin        := conversionValid
  lhsMul.io.op             := 0.U
  lhsMul.io.a              := lhsToFloat.io.out
  lhsMul.io.b              := recFNFromFN(8, 24, ratioStage1)
  lhsMul.io.c              := recFNFromFN(8, 24, 0.U)
  lhsMul.io.roundingMode   := round_near_even
  lhsMul.io.detectTininess := tininess_afterRounding.asUInt
  rhsMul.io.validin        := conversionValid
  rhsMul.io.op             := 0.U
  rhsMul.io.a              := rhsToFloat.io.out
  rhsMul.io.b              := recFNFromFN(8, 24, rhsRatioStage1)
  rhsMul.io.c              := recFNFromFN(8, 24, 0.U)
  rhsMul.io.roundingMode   := round_near_even
  rhsMul.io.detectTininess := tininess_afterRounding.asUInt

  add.io.validin          := lhsMul.io.validout
  add.io.subOp            := false.B
  add.io.a                := rawFloatFromRecFN(8, 24, lhsMul.io.out)
  add.io.b                := rawFloatFromRecFN(8, 24, rhsMul.io.out)
  add.io.roundingMode     := round_near_even
  round.io.invalidExc     := add.io.invalidExc
  round.io.infiniteExc    := false.B
  round.io.in             := addResult
  round.io.roundingMode   := round_near_even
  round.io.detectTininess := tininess_afterRounding
  toInt.io.in             := roundedResult
  toInt.io.roundingMode   := round_near_even
  toInt.io.signedOut      := true.B
  val result = Mux(relu && toInt.io.out(7), 0.U, toInt.io.out)

  blink.cmdReq.ready        := state === idle
  blink.cmdResp.valid       := state === complete
  blink.cmdResp.bits.rob_id := robId
  blink.status.idle         := state === idle
  blink.status.running      := state =/= idle && state =/= complete

  for (port <- 0 until 2) {
    blink.bankRead(port).rob_id           := robId
    blink.bankRead(port).bank_id          := Mux(port.U === 0.U, lhsBank, rhsBank)
    blink.bankRead(port).group_id         := bases(port) + group
    blink.bankRead(port).io.req.valid     := false.B
    blink.bankRead(port).io.req.bits.addr := row
    blink.bankRead(port).io.resp.ready    := false.B
  }
  blink.bankWrite(0).rob_id := robId
  blink.bankWrite(0).bank_id          := outputBank
  blink.bankWrite(0).group_id         := bases(2) + group
  blink.bankWrite(0).io.req.valid     := false.B
  blink.bankWrite(0).io.req.bits.addr := row
  blink.bankWrite(0).io.req.bits.data := Cat(outputWord.reverse)
  blink.bankWrite(0).io.req.bits.mask := VecInit(Seq.fill(16)(true.B))
  blink.bankWrite(0).io.resp.ready    := false.B

  switch(state) {
    is(idle) {
      when(blink.cmdReq.fire) {
        val cmd = blink.cmdReq.bits
        assert(
          cmd.funct === addFunct.U || cmd.funct === reluFunct.U,
          "Int8AddBall received an unknown funct7"
        )
        assert(
          positiveFinite(cmd.rs2(31, 0)) && positiveFinite(cmd.rs2(63, 32)),
          "Int8AddBall ratios must be finite and positive"
        )
        assert(
          cmd.op1_col =/= 0.U && cmd.op1_col === cmd.op2_col && cmd.op1_col === cmd.wr_col,
          "Int8AddBall bank groups must match"
        )
        assert(
          cmd.op1_col <= mp.bankNum.U,
          "Int8AddBall bank groups exceed physical banks"
        )
        assert(
          cmd.op1_bank =/= cmd.op2_bank && cmd.op1_bank =/= cmd.wr_bank && cmd.op2_bank =/= cmd.wr_bank,
          "Int8AddBall banks must be distinct"
        )
        assert(
          cmd.iter =/= 0.U && cmd.iter <= mp.bankEntries.U,
          "Int8AddBall iter must fit one physical bank"
        )
        robId        := blink.cmdReq.bits.rob_id
        lhsBank      := cmd.op1_bank
        rhsBank      := cmd.op2_bank
        outputBank   := cmd.wr_bank
        groups       := cmd.op1_col
        group        := 0.U
        bases        := VecInit(cmd.op1_group, cmd.op2_group, cmd.wr_group)
        iter         := cmd.iter
        row          := 0.U
        lhsRatio     := cmd.rs2(31, 0)
        rhsRatio     := cmd.rs2(63, 32)
        relu         := cmd.funct === reluFunct.U
        lhsRequested := false.B
        rhsRequested := false.B
        lhsReceived  := false.B
        rhsReceived  := false.B
        lane         := 0.U
        state        := waitForChannels
      }
    }
    is(waitForChannels) {
      state := readRequest
    }
    is(readRequest) {
      blink.bankRead(0).io.req.valid                   := !lhsRequested
      blink.bankRead(1).io.req.valid                   := !rhsRequested
      when(blink.bankRead(0).io.req.fire)(lhsRequested := true.B)
      when(blink.bankRead(1).io.req.fire)(rhsRequested := true.B)
      when((lhsRequested || blink.bankRead(0).io.req.fire) && (rhsRequested || blink.bankRead(1).io.req.fire)) {
        state := readResponse
      }
    }
    is(readResponse) {
      blink.bankRead(0).io.resp.ready := !lhsReceived
      blink.bankRead(1).io.resp.ready := !rhsReceived
      when(blink.bankRead(0).io.resp.fire) { lhsWord := blink.bankRead(0).io.resp.bits.data; lhsReceived := true.B }
      when(blink.bankRead(1).io.resp.fire) { rhsWord := blink.bankRead(1).io.resp.bits.data; rhsReceived := true.B }
      when((lhsReceived || blink.bankRead(0).io.resp.fire) && (rhsReceived || blink.bankRead(1).io.resp.fire)) {
        lane      := 0.U
        completed := 0.U
        state     := calculate
      }
    }
    is(calculate) {
      when(lane === 15.U) {
        state := calculateDrain
      }.otherwise {
        lane := lane + 1.U
      }
    }
    is(calculateDrain) {}
    is(writeRequest) {
      blink.bankWrite(0).io.req.valid            := true.B
      when(blink.bankWrite(0).io.req.fire)(state := writeResponse)
    }
    is(writeResponse) {
      blink.bankWrite(0).io.resp.ready := true.B
      when(blink.bankWrite(0).io.resp.fire) {
        lhsRequested := false.B
        rhsRequested := false.B
        lhsReceived  := false.B
        rhsReceived  := false.B
        when(row +& 1.U === iter) {
          row := 0.U
          when(group +& 1.U === groups)(state := complete)
            .otherwise { group := group + 1.U; state := readRequest }
        }.otherwise { row := row + 1.U; state := readRequest }
      }
    }
    is(complete) {
      when(blink.cmdResp.fire)(state := idle)
    }
  }

  when((state === calculate || state === calculateDrain) && resultValid) {
    outputWord(roundedLane.bits) := result
    completed                    := completed + 1.U
    when(completed === 15.U) {
      state := writeRequest
    }
  }
}

object Int8AddBall extends BallFactory {

  def instantiate(
    fp: FrontendParam,
    mp: MemDomainParam,
    bp: BallDomainParam,
    id: Int
  ): BallPorts =
    BallPorts(Instantiate(new Int8AddBall(fp, mp, bp, id)).blink, None)

}
