package examples.balls.f32add

import framework.acc.balldomain.bbus.{BallFactory, BallPorts}
import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.acc.frontend.configs.FrontendParam
import buckyball.balls.f32add.{Params => ParametersProto}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.acc.balldomain.ball.blink.{BallCommand, BlinkIO, HasBlink}

@instantiable
class F32AddBall(
  fp: FrontendParam,
  mp: MemDomainParam,
  bp: BallDomainParam,
  id: Int)
    extends Module
    with HasBlink {
  val ball = bp.balls.find(_.id == id).get

  ball.parameters.unpack(classOf[ParametersProto])

  require(ball.inBW == 2 && ball.outBW == 1)
  require(
    mp.bankWidth == 128 && mp.bankMaskLen == 16
  )

  @public
  val blink = IO(new BlinkIO(fp, mp, bp, id))

  val idle :: channels :: reading :: calculate :: writing :: acknowledge :: complete :: Nil = Enum(7)
  val state                                                                                 = RegInit(idle)

  val issue = Reg(new BallCommand(fp, mp))

  val row         = RegInit(0.U(16.W))
  val requested   = RegInit(VecInit(Seq.fill(2)(false.B)))
  val received    = RegInit(VecInit(Seq.fill(2)(false.B)))
  val words       = Reg(Vec(2, UInt(128.W)))
  val output      = Reg(UInt(128.W))
  val first       = issue.rs2(5)
  val sourceGroup = issue.rs2(4, 0)
  val add         = Instantiate(new F32AddRow)
  add.io.source      := words(0)
  add.io.accumulator := Mux(first, 0.U(128.W), words(1))

  blink.cmdReq.ready        := state === idle
  blink.cmdResp.valid       := state === complete
  blink.cmdResp.bits.rob_id := issue.rob_id
  blink.status.idle         := state === idle
  blink.status.running      := state =/= idle

  for (port <- 0 until 2) {
    val read   = blink.bankRead(port)
    val active = if (port == 0) true.B else !first
    read.rob_id                            := issue.rob_id
    read.bank_id                           := (if (port == 0) issue.op1_bank else issue.op2_bank)
    read.group_id                          := (if (port == 0) issue.op1_group + sourceGroup else issue.op2_group)
    read.io.req.valid                      := state === reading && active && !requested(port)
    read.io.req.bits.addr                  := row
    read.io.resp.ready                     := state === reading && active && requested(port) && !received(port)
    when(read.io.req.fire)(requested(port) := true.B)
    when(read.io.resp.fire) {
      words(port)    := read.io.resp.bits.data
      received(port) := true.B
    }
  }

  val write = blink.bankWrite(0)
  write.rob_id           := issue.rob_id
  write.bank_id          := issue.wr_bank
  write.group_id         := issue.wr_group
  write.io.req.valid     := state === writing
  write.io.req.bits.addr := row
  write.io.req.bits.data := output
  write.io.req.bits.mask := VecInit(Seq.fill(16)(true.B))
  write.io.resp.ready    := state === acknowledge

  when(blink.cmdReq.fire) {
    val cmd = blink.cmdReq.bits
    assert(cmd.rs2(63, 6) === 0.U, "F32ADD reserved rs2 bits")
    assert(
      cmd.op1_bank =/= cmd.op2_bank && cmd.op1_bank =/= cmd.wr_bank && cmd.op2_bank =/= cmd.wr_bank,
      "F32ADD banks must differ"
    )
    assert(
      cmd.rs2(4, 0) < cmd.op1_col && cmd.op2_col === 1.U && cmd.wr_col === 1.U,
      "F32ADD source group must exist and accumulators must have one group"
    )
    for (bank <- Seq(cmd.op1_bank, cmd.op2_bank, cmd.wr_bank)) {
      val shared = bank >= fp.shared_vbank_base.U
      assert(
        !shared || mp.sharedEnable.B,
        "F32ADD shared memory is disabled"
      )
      val depth  = Mux(shared, mp.sharedBankEntries.U, mp.bankEntries.U)
      assert(cmd.iter =/= 0.U && cmd.iter <= depth, "F32ADD row span exceeds bank")
    }
    issue := blink.cmdReq.bits
    row                 := 0.U
    requested.foreach(_ := false.B)
    received.foreach(_  := false.B)
    state               := channels
  }
  when(state === channels)(state                                         := reading)
  when(state === reading && received(0) && (first || received(1)))(state := calculate)
  when(state === calculate) {
    output := add.io.sum
    state  := writing
  }
  when(write.io.req.fire)(state                                          := acknowledge)
  when(write.io.resp.fire) {
    assert(write.io.resp.bits.ok, "F32ADD bank write failed")
    when(row +& 1.U === issue.iter) {
      state := complete
    }.otherwise {
      row                 := row + 1.U
      requested.foreach(_ := false.B)
      received.foreach(_  := false.B)
      state               := reading
    }
  }
  when(blink.cmdResp.fire)(state                                         := idle)
}

object F32AddBall extends BallFactory {

  def instantiate(
    fp: FrontendParam,
    mp: MemDomainParam,
    bp: BallDomainParam,
    id: Int
  ): BallPorts =
    BallPorts(Instantiate(new F32AddBall(fp, mp, bp, id)).blink, None)

}
