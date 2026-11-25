package framework.ball.bbus

import framework.generic.contract.ball.BallHostParam

import framework.generic.contract.blink.BallResponse

import framework.generic.queue.Queue
import framework.generic.arbiter.{Arbiter, RRArbiter}

import framework.ball.sub_rob.Expansion
import framework.ball.blink.BlinkIO

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.ball.bbus.channels.Channels
import framework.generic.contract.decoder.DISA.BALL_INIT_BITPAT
import framework.ball.mmio_bank.{MmioRead, MmioWriteReq}
import framework.generic.contract.configs.BallDomainParam
import framework.generic.contract.memdomain.BankLayout
import framework.generic.contract.memdomain.BankLayoutRequest
import framework.generic.contract.memdomain.BankPort
import framework.generic.contract.blink.{BallCommand, BlinkParam}
import framework.generic.contract.configs.BallRouteParam
import framework.ball.sub_rob.ExpansionParam
import framework.generic.contract.bbus.BallRequest

/** Interconnect for the Ball instances listed in the BallDomain configuration. */
@instantiable
class BBus(
  host:      BallHostParam,
  bp:        BallDomainParam,
  factories: Map[String, BallFactory])
    extends Module {
  val numBalls = bp.balls.size
  // All a Ball is built from, with its own route.
  val link     = host.link

  @public
  val io = IO(new Bundle {
    // The thread (or hart) the accelerator serves: a strap, so that every
    // thread of a hart shares one module.
    val ownerId        = Input(UInt(64.W))
    val mmioRead       = Vec(bp.balls.map(_.mmioReadBW).sum, Flipped(new MmioRead(link)))
    val mmioWrite      = Vec(bp.balls.map(_.mmioWriteBW).sum, Decoupled(new MmioWriteReq(link)))
    val traceStarted   = Output(Vec(numBalls, Valid(UInt(host.link.robIdBits.W))))
    val traceCompleted = Output(Vec(numBalls, Valid(UInt(host.link.robIdBits.W))))
    // Frontend commands still need their group counts from the bank layout.
    val cmdReq         = Flipped(Decoupled(new BallRequest(host.link)))
    val cmdResp        = Decoupled(new BallResponse(link))
    val layoutRequest  = Decoupled(new BankLayoutRequest(host.link.bankIdBits))
    val layoutResponse = Flipped(Decoupled(new BankLayout(host.link.groupCountBits)))
    val banks          = Vec(host.bankRequestPorts, new BankPort(host.link))
    // GP operator commands arrive complete and complete back to the operator.
    val operatorReq    = Option.when(host.operators)(Flipped(Decoupled(new BallRequest(host.link))))
    val operatorResp   = Option.when(host.operators)(Decoupled(new BallResponse(link)))
  })

  val channels: Instance[Channels] = Instantiate(new Channels(host, bp))
  channels.io.ownerId := io.ownerId
  val instructionIds = Reg(Vec(numBalls, UInt(64.W)))
  channels.io.instructionIds := instructionIds
  io.banks <> channels.io.mem_req

  // Apply BALL_INIT on the cycle after the command handshake.  Driving reset
  // directly from cmd.fire would feed the Ball's reset-dependent ready signal
  // back into that same handshake.
  val ballBootReset = RegInit(VecInit(Seq.fill(numBalls)(false.B)))

  val instances = bp.balls.zipWithIndex.map { case (ball, index) =>
    withReset(reset.asBool || ballBootReset(index)) {
      factories(ball.component).instantiate(link, ball, Option.when(host.subRobEnable)(ExpansionParam(host, bp)))
    }
  }

  val balls    = instances.map(_.blink)
  val expanded = instances.flatMap(_.expansion)
  require(expanded.size <= 1, "One instruction expander is supported per BallDomain")
  require(expanded.nonEmpty == host.subRobEnable)

  @public
  val expansion: Option[Expansion] = expanded.headOption.map { source =>
    val port = IO(new Expansion(host, bp))
    port <> source
    port
  }

  balls.flatMap(_.mmioRead).zip(io.mmioRead).foreach { case (source, port) => port <> source }
  balls.flatMap(_.mmioWrite).zip(io.mmioWrite).foreach { case (source, port) => port <> source }

  val pending = Instantiate(new Queue(new BallRequest(host.link), 4))
  io.layoutRequest.valid         := io.cmdReq.valid && pending.io.enq.ready
  pending.io.enq.valid           := io.cmdReq.valid && io.layoutRequest.ready
  pending.io.enq.bits            := io.cmdReq.bits
  io.cmdReq.ready                := pending.io.enq.ready && io.layoutRequest.ready
  io.layoutRequest.bits.banks(0) := io.cmdReq.bits.cmd.op1_bank
  io.layoutRequest.bits.banks(1) := io.cmdReq.bits.cmd.op2_bank
  io.layoutRequest.bits.banks(2) := io.cmdReq.bits.cmd.wr_bank
  io.layoutRequest.bits.enabled  := io.cmdReq.bits.enabled
  val external = Wire(Decoupled(new BallRequest(host.link)))
  external.valid            := pending.io.deq.valid && io.layoutResponse.valid
  external.bits             := pending.io.deq.bits
  external.bits.cmd.op1_col := io.layoutResponse.bits.groups(0)
  external.bits.cmd.op2_col := io.layoutResponse.bits.groups(1)
  external.bits.cmd.wr_col  := io.layoutResponse.bits.groups(2)
  pending.io.deq.ready      := external.ready && io.layoutResponse.valid
  io.layoutResponse.ready   := external.ready && pending.io.deq.valid

  val command         = Wire(Decoupled(new BallRequest(host.link)))
  val fromOperator    = WireDefault(false.B)
  val operatorOwner   = RegInit(VecInit(Seq.fill(numBalls)(false.B)))
  val operatorPending = WireDefault(false.B)
  if (host.operators) {
    val arbiter = Instantiate(new RRArbiter(new BallRequest(host.link), 2))
    arbiter.io.in(0) <> external
    arbiter.io.in(1) <> io.operatorReq.get
    command <> arbiter.io.out
    fromOperator := arbiter.io.chosen === 1.U
    assert(PopCount(operatorOwner) <= 1.U, "GP domain issues only one outstanding Ball operation")
  } else {
    command <> external
  }

  val isBallInit    = command.bits.cmd.funct === BALL_INIT_BITPAT
  val initPending   = RegInit(VecInit(Seq.fill(numBalls)(false.B)))
  val initResp      = Reg(Vec(numBalls, new BallResponse(link)))
  val targetMatches = VecInit(bp.balls.map(b => command.bits.target === b.id.U))
  val responses     = Instantiate(new Arbiter(new BallResponse(link), numBalls))
  val operatorDone  = Wire(Vec(numBalls, Valid(new BallResponse(link))))

  for (i <- 0 until numBalls) {
    channels.io.active(i) := balls(i).status.running

    val targetMatch = targetMatches(i)
    balls(i).cmdReq.valid := command.valid && !isBallInit && !initPending(i) && targetMatch
    balls(i).cmdReq.bits  := command.bits.cmd

    val done = Mux(initPending(i), true.B, balls(i).cmdResp.valid)
    val resp = Mux(initPending(i), initResp(i), balls(i).cmdResp.bits)
    responses.io.in(i).valid := !operatorOwner(i) && done
    responses.io.in(i).bits  := resp
    operatorDone(i).valid    := operatorOwner(i) && done
    operatorDone(i).bits     := resp

    val accepted = Mux(operatorOwner(i), io.operatorResp.map(_.ready).getOrElse(false.B), responses.io.in(i).ready)
    balls(i).cmdResp.ready := !initPending(i) && accepted
    when(done && accepted) {
      initPending(i)   := false.B
      operatorOwner(i) := false.B
    }

    when(command.fire && targetMatch) {
      instructionIds(i) := command.bits.instId
      operatorOwner(i)  := fromOperator
    }
    ballBootReset(i) := command.fire && isBallInit && targetMatch
    when(command.fire && isBallInit && targetMatch) {
      initPending(i)     := true.B
      initResp(i).rob_id := command.bits.cmd.rob_id
    }

    io.traceStarted(i).valid   := command.fire && targetMatch
    io.traceStarted(i).bits    := command.bits.cmd.rob_id
    io.traceCompleted(i).valid := done && accepted
    io.traceCompleted(i).bits  := resp.rob_id
  }

  io.operatorResp.foreach { port =>
    port.valid := operatorDone.map(_.valid).reduce(_ || _)
    port.bits  := Mux1H(operatorDone.map(_.valid), operatorDone.map(_.bits))
  }

  command.ready :=
    VecInit((0 until numBalls).map(i => balls(i).cmdReq.ready && !initPending(i) && targetMatches(i))).asUInt.orR

  io.cmdResp <> responses.io.out

  // Ball bank ports occupy consecutive Channel slots in configuration order.
  channels.io.bankRead.zip(balls.flatMap(_.bankRead)).foreach { case (channel, port) => channel <> port }
  channels.io.bankWrite.zip(balls.flatMap(_.bankWrite)).foreach { case (channel, port) => channel <> port }
}

case class BallPorts(blink: BlinkIO, expansion: Option[Expansion])

/**
 * A Ball is built from its link parameters and its route. `expansion` is there for a Ball that
 * expands instructions into the SubROB, the only kind that sees host parameters.
 */
trait BallFactory {

  def instantiate(
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts

}
