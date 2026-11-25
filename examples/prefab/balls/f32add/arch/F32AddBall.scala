package examples.balls.f32add

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam
import framework.ball.sub_rob.ExpansionParam

import framework.ball.bbus.{BallFactory, BallPorts}
import framework.generic.queue.{Queue => ResponseQueue}
import buckyball.balls.f32add.{Params => ParametersProto}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.ball.blink.{BlinkIO, HasBlink}
import framework.generic.contract.blink.BallCommand

@instantiable
class F32AddBall(
  link:  BlinkParam,
  route: BallRouteParam)
    extends Module
    with HasBlink {
  val ball = route

  ball.parameters.unpack(classOf[ParametersProto])

  require(ball.inBW == 2 && ball.outBW == 1)
  require(
    link.bankWidth == 128 && link.bankMaskLen == 16
  )

  @public
  val blink = IO(new BlinkIO(link, route))

  val idle :: channels :: running :: complete :: Nil = Enum(4)
  val state                                          = RegInit(idle)

  val issue = Reg(new BallCommand(link))

  val requested    = RegInit(VecInit(Seq.fill(2)(0.U(16.W))))
  val credits      = RegInit(VecInit(Seq.fill(2)(0.U(2.W))))
  val written      = RegInit(0.U(16.W))
  val acknowledged = RegInit(0.U(16.W))
  val operands     = Seq.fill(2)(Instantiate(new ResponseQueue(UInt(128.W), 2)))
  val results      = Instantiate(new ResponseQueue(UInt(128.W), 2))
  val first        = issue.rs2(5)
  val sourceGroup  = issue.rs2(4, 0)
  val add          = Instantiate(new F32AddRow)
  add.io.source            := operands(0).io.deq.bits
  add.io.accumulator       := Mux(first, 0.U(128.W), operands(1).io.deq.bits)
  results.io.enq.valid     := state === running && operands(0).io.deq.valid &&
    (first || operands(1).io.deq.valid)
  results.io.enq.bits      := add.io.sum
  operands(0).io.deq.ready := results.io.enq.ready && (first || operands(1).io.deq.valid) && state === running
  operands(1).io.deq.ready := results.io.enq.ready && operands(0).io.deq.valid && !first && state === running

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
    // Credits include both in-flight reads and queued operands. Each port
    // preserves row order while independent bank stalls remain decoupled.
    read.io.req.valid                      := state === running && active && requested(port) < issue.iter && credits(port) < 2.U
    read.io.req.bits.addr                  := requested(port)
    operands(port).io.enq.valid            := read.io.resp.valid && state === running
    operands(port).io.enq.bits             := read.io.resp.bits.data
    read.io.resp.ready                     := operands(port).io.enq.ready && state === running
    when(read.io.req.fire)(requested(port) := requested(port) + 1.U)
    when(read.io.req.fire =/= operands(port).io.deq.fire) {
      credits(port) := Mux(read.io.req.fire, credits(port) + 1.U, credits(port) - 1.U)
    }
  }

  val write = blink.bankWrite(0)
  write.rob_id           := issue.rob_id
  write.bank_id          := issue.wr_bank
  write.group_id         := issue.wr_group
  write.io.req.valid     := state === running && results.io.deq.valid
  write.io.req.bits.addr := written
  write.io.req.bits.data := results.io.deq.bits
  write.io.req.bits.mask := VecInit(Seq.fill(16)(true.B))
  write.io.resp.ready    := state === running
  results.io.deq.ready   := write.io.req.ready && state === running

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
    assert(cmd.iter =/= 0.U && cmd.iter <= link.bankEntries.U, "F32ADD row span exceeds bank")
    issue               := blink.cmdReq.bits
    requested.foreach(_ := 0.U)
    credits.foreach(_   := 0.U)
    written             := 0.U
    acknowledged        := 0.U
    state               := channels
  }
  when(state === channels)(state  := running)
  when(write.io.req.fire)(written := written + 1.U)
  when(write.io.resp.fire) {
    assert(write.io.resp.bits.ok, "F32ADD bank write failed")
    acknowledged := acknowledged + 1.U
    when(acknowledged +& 1.U === issue.iter) {
      state := complete
    }
  }
  when(blink.cmdResp.fire)(state  := idle)
}

object F32AddBall extends BallFactory {

  def instantiate(
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts =
    BallPorts(Instantiate(new F32AddBall(link, route)).blink, None)

}
