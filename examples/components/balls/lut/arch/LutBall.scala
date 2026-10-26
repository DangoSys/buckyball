package examples.balls.lut

import framework.acc.balldomain.bbus.{BallFactory, BallPorts}
import buckyball.balls.lut.{Params => ParametersProto}

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import chisel3.util._
import framework.acc.balldomain.ball.blink.{BallStatus, BlinkIO, HasBlink}

@instantiable
class LutBall(
  fp: FrontendParam,
  mp: MemDomainParam,
  bp: BallDomainParam,
  id: Int)
    extends Module
    with HasBlink {
  val ball = bp.balls.find(_.id == id).get

  ball.parameters.unpack(classOf[ParametersProto])

  require(mp.bankWidth == 128, "LutBall requires 128-bit bank rows")
  require(mp.bankEntries >= 16, "LutBall requires sixteen LUT rows")
  require(ball.inBW == 2, "LutBall requires inBW=2")
  require(ball.outBW == 1, "LutBall requires outBW=1")

  @public
  val blink = IO(new BlinkIO(fp, mp, bp, id))

  val idle :: inputReq :: inputResp :: lutReq :: lutResp :: writeReq :: writeResp :: complete :: Nil =
    Enum(8)
  val state                                                                                          = RegInit(idle)
  val robId                                                                                          = RegInit(0.U(log2Up(fp.rob_entries).W))
  val inputBank                                                                                      = RegInit(0.U(fp.bank_id_len.W))
  val lutBank                                                                                        = RegInit(0.U(fp.bank_id_len.W))
  val outputBank                                                                                     = RegInit(0.U(fp.bank_id_len.W))
  val bases                                                                                          = Reg(Vec(3, UInt(mp.groupIdWidth.W)))
  val iter                                                                                           = RegInit(0.U(fp.iter_len.W))
  val inputRow                                                                                       = RegInit(0.U(log2Ceil(mp.bankEntries).W))
  val inputData                                                                                      = Reg(UInt(128.W))
  val laneTable                                                                                      = RegInit(false.B)
  val lookupLane                                                                                     = RegInit(0.U(4.W))
  val outputWord                                                                                     = Reg(Vec(16, UInt(8.W)))

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
  blink.bankRead(0).bank_id := inputBank
  blink.bankRead(1).bank_id  := lutBank
  blink.bankRead(1).group_id := bases(1) + Mux(laneTable, lookupLane >> 2, 0.U)

  blink.bankWrite(0).rob_id           := robId
  blink.bankWrite(0).bank_id          := outputBank
  blink.bankWrite(0).group_id         := bases(2)
  blink.bankWrite(0).io.req.valid     := false.B
  blink.bankWrite(0).io.req.bits.addr := inputRow
  blink.bankWrite(0).io.req.bits.data := Cat(outputWord.reverse)
  blink.bankWrite(0).io.req.bits.mask := VecInit(
    Seq.fill(mp.bankMaskLen)(true.B)
  )
  blink.bankWrite(0).io.resp.ready    := false.B

  switch(state) {
    is(idle) {
      when(blink.cmdReq.fire) {
        val cmd = blink.cmdReq.bits

        assert(cmd.rs2 === 0.U, "LutBall reserves rs2")
        assert(
          cmd.iter > 0.U && cmd.iter <= mp.bankEntries.U,
          "LutBall iter must fit in one bank"
        )
        assert(
          cmd.op1_col === 1.U && (cmd.op2_col === 1.U || cmd.op2_col === 4.U) && cmd.wr_col === 1.U,
          "LutBall requires col=1 input/output and col=1 or col=4 table"
        )
        assert(
          cmd.op1_bank =/= cmd.op2_bank && cmd.op1_bank =/= cmd.wr_bank &&
            cmd.op2_bank =/= cmd.wr_bank,
          "LutBall banks must be distinct"
        )
        robId      := blink.cmdReq.bits.rob_id
        inputBank  := cmd.op1_bank
        lutBank    := cmd.op2_bank
        outputBank := cmd.wr_bank
        bases      := VecInit(cmd.op1_group, cmd.op2_group, cmd.wr_group)
        iter       := cmd.iter
        inputRow   := 0.U
        laneTable  := cmd.op2_col === 4.U
        lookupLane := 0.U
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
        inputData  := blink.bankRead(0).io.resp.bits.data
        lookupLane := 0.U
        state      := lutReq
      }
    }
    is(lutReq) {
      val inputByte = (inputData >> (lookupLane << 3))(7, 0)
      blink.bankRead(1).io.req.valid            := true.B
      blink.bankRead(1).io.req.bits.addr        := Mux(
        laneTable,
        Cat(lookupLane(1, 0), inputByte(7, 4)),
        inputByte(7, 4)
      )
      when(blink.bankRead(1).io.req.fire)(state := lutResp)
    }
    is(lutResp) {
      val inputByte = (inputData >> (lookupLane << 3))(7, 0)
      blink.bankRead(1).io.resp.ready := true.B
      when(blink.bankRead(1).io.resp.fire) {
        outputWord(lookupLane) :=
          (blink.bankRead(1).io.resp.bits.data >> (inputByte(3, 0) << 3))(7, 0)
        when(lookupLane === 15.U)(state := writeReq)
          .otherwise { lookupLane := lookupLane + 1.U; state := lutReq }
      }
    }
    is(writeReq) {
      blink.bankWrite(0).io.req.valid            := true.B
      when(blink.bankWrite(0).io.req.fire)(state := writeResp)
    }
    is(writeResp) {
      blink.bankWrite(0).io.resp.ready := true.B
      when(blink.bankWrite(0).io.resp.fire) {
        when(inputRow === iter - 1.U)(state := complete)
          .otherwise { inputRow := inputRow + 1.U; state := inputReq }
      }
    }
    is(complete) {
      when(blink.cmdResp.fire)(state := idle)
    }
  }
}

object LutBall extends BallFactory {

  def instantiate(
    fp: FrontendParam,
    mp: MemDomainParam,
    bp: BallDomainParam,
    id: Int
  ): BallPorts =
    BallPorts(Instantiate(new LutBall(fp, mp, bp, id)).blink, None)

}
