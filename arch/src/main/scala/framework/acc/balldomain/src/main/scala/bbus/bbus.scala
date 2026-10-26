package framework.acc.balldomain.bbus

import framework.lib.queue.Queue
import framework.lib.arbiter.{Arbiter, RRArbiter}

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.balldomain.sub_rob.Expansion
import framework.acc.balldomain.ball.blink.BlinkIO
import framework.acc.frontend.configs.FrontendParam
import framework.acc.gpdomain.configs.GpDomainParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.memdomain.{BankLayout, BankLayoutRequest, BankPort}
import framework.acc.balldomain.bbus.channels.Channels
import framework.acc.balldomain.ball.blink.{BallCommand, BallResponse}
import framework.acc.balldomain.decoder.DISA.BALL_INIT_BITPAT
import framework.acc.balldomain.mmio_bank.{MmioRead, MmioWriteReq}

/** A Ball command routed to `target`; `enabled` marks which of op1, op2 and wr name a bank. */
class BallRequest(fp: FrontendParam, mp: MemDomainParam) extends Bundle {
  val target  = UInt(5.W)
  val instId  = UInt(64.W) // Observation identity; never controls scheduling.
  val enabled = Vec(3, Bool())
  val cmd     = new BallCommand(fp, mp)
}

/** Interconnect for the Ball instances listed in the BallDomain configuration. */
@instantiable
class BBus(
  fp:        FrontendParam,
  mp:        MemDomainParam,
  gp:        GpDomainParam,
  bp:        BallDomainParam,
  factories: Map[String, BallFactory])
    extends Module {
  val numBalls = bp.balls.size

  @public
  val io = IO(new Bundle {
    val mmioRead       = Vec(bp.balls.map(_.mmioReadBW).sum, Flipped(new MmioRead(mp)))
    val mmioWrite      = Vec(bp.balls.map(_.mmioWriteBW).sum, Decoupled(new MmioWriteReq(mp)))
    val traceStarted   = Output(Vec(numBalls, Valid(UInt(log2Ceil(fp.rob_entries).W))))
    val traceCompleted = Output(Vec(numBalls, Valid(UInt(log2Ceil(fp.rob_entries).W))))
    // Frontend commands still need their group counts from the bank layout.
    val cmdReq         = Flipped(Decoupled(new BallRequest(fp, mp)))
    val cmdResp        = Decoupled(new BallResponse(fp))
    val layoutRequest  = Decoupled(new BankLayoutRequest(fp))
    val layoutResponse = Flipped(Decoupled(new BankLayout(mp)))
    val banks          = Vec(mp.bankRequestPorts, new BankPort(fp, mp))
    // GP kernel commands arrive complete and complete back to the kernel.
    val kernelReq      = Option.when(gp.enable)(Flipped(Decoupled(new BallRequest(fp, mp))))
    val kernelResp     = Option.when(gp.enable)(Decoupled(new BallResponse(fp)))
  })

  val channels: Instance[Channels] = Instantiate(new Channels(fp, mp, bp))
  val instructionIds = Reg(Vec(numBalls, UInt(64.W)))
  channels.io.instructionIds := instructionIds
  io.banks <> channels.io.mem_req

  // Apply BALL_INIT on the cycle after the command handshake.  Driving reset
  // directly from cmd.fire would feed the Ball's reset-dependent ready signal
  // back into that same handshake.
  val ballBootReset = RegInit(VecInit(Seq.fill(numBalls)(false.B)))

  val instances = bp.balls.zipWithIndex.map { case (ball, index) =>
    withReset(reset.asBool || ballBootReset(index)) {
      factories(ball.component).instantiate(fp, mp, bp, ball.id)
    }
  }

  val balls    = instances.map(_.blink)
  val expanded = instances.flatMap(_.expansion)
  require(expanded.size <= 1, "One instruction expander is supported per BallDomain")
  require(expanded.nonEmpty == fp.sub_rob_enable)

  @public
  val expansion: Option[Expansion] = expanded.headOption.map { source =>
    val port = IO(new Expansion(fp, mp, bp))
    port <> source
    port
  }

  balls.flatMap(_.mmioRead).zip(io.mmioRead).foreach { case (source, port) => port <> source }
  balls.flatMap(_.mmioWrite).zip(io.mmioWrite).foreach { case (source, port) => port <> source }

  val pending = Instantiate(new Queue(new BallRequest(fp, mp), 4))
  io.layoutRequest.valid         := io.cmdReq.valid && pending.io.enq.ready
  pending.io.enq.valid           := io.cmdReq.valid && io.layoutRequest.ready
  pending.io.enq.bits            := io.cmdReq.bits
  io.cmdReq.ready                := pending.io.enq.ready && io.layoutRequest.ready
  io.layoutRequest.bits.banks(0) := io.cmdReq.bits.cmd.op1_bank
  io.layoutRequest.bits.banks(1) := io.cmdReq.bits.cmd.op2_bank
  io.layoutRequest.bits.banks(2) := io.cmdReq.bits.cmd.wr_bank
  io.layoutRequest.bits.enabled  := io.cmdReq.bits.enabled
  val external = Wire(Decoupled(new BallRequest(fp, mp)))
  external.valid            := pending.io.deq.valid && io.layoutResponse.valid
  external.bits             := pending.io.deq.bits
  external.bits.cmd.op1_col := io.layoutResponse.bits.groups(0)
  external.bits.cmd.op2_col := io.layoutResponse.bits.groups(1)
  external.bits.cmd.wr_col  := io.layoutResponse.bits.groups(2)
  pending.io.deq.ready      := external.ready && io.layoutResponse.valid
  io.layoutResponse.ready   := external.ready && pending.io.deq.valid

  val command       = Wire(Decoupled(new BallRequest(fp, mp)))
  val fromKernel    = WireDefault(false.B)
  val kernelOwner   = RegInit(VecInit(Seq.fill(numBalls)(false.B)))
  val kernelPending = WireDefault(false.B)
  if (gp.enable) {
    val arbiter = Instantiate(new RRArbiter(new BallRequest(fp, mp), 2))
    arbiter.io.in(0) <> external
    arbiter.io.in(1) <> io.kernelReq.get
    command <> arbiter.io.out
    fromKernel := arbiter.io.chosen === 1.U
    assert(PopCount(kernelOwner) <= 1.U, "GP domain issues only one outstanding Ball operation")
  } else {
    command <> external
  }

  val isBallInit    = command.bits.cmd.funct === BALL_INIT_BITPAT
  val initPending   = RegInit(VecInit(Seq.fill(numBalls)(false.B)))
  val initResp      = Reg(Vec(numBalls, new BallResponse(fp)))
  val targetMatches = VecInit(bp.balls.map(b => command.bits.target === b.id.U))
  val responses     = Instantiate(new Arbiter(new BallResponse(fp), numBalls))
  val kernelDone    = Wire(Vec(numBalls, Valid(new BallResponse(fp))))

  for (i <- 0 until numBalls) {
    channels.io.active(i) := balls(i).status.running

    val targetMatch = targetMatches(i)
    balls(i).cmdReq.valid := command.valid && !isBallInit && !initPending(i) && targetMatch
    balls(i).cmdReq.bits  := command.bits.cmd

    val done = Mux(initPending(i), true.B, balls(i).cmdResp.valid)
    val resp = Mux(initPending(i), initResp(i), balls(i).cmdResp.bits)
    responses.io.in(i).valid := !kernelOwner(i) && done
    responses.io.in(i).bits  := resp
    kernelDone(i).valid      := kernelOwner(i) && done
    kernelDone(i).bits       := resp

    val accepted = Mux(kernelOwner(i), io.kernelResp.map(_.ready).getOrElse(false.B), responses.io.in(i).ready)
    balls(i).cmdResp.ready := !initPending(i) && accepted
    when(done && accepted) {
      initPending(i) := false.B
      kernelOwner(i) := false.B
    }

    when(command.fire && targetMatch) {
      instructionIds(i) := command.bits.instId
      kernelOwner(i)    := fromKernel
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

  io.kernelResp.foreach { port =>
    port.valid := kernelDone.map(_.valid).reduce(_ || _)
    port.bits  := Mux1H(kernelDone.map(_.valid), kernelDone.map(_.bits))
  }

  command.ready :=
    VecInit((0 until numBalls).map(i => balls(i).cmdReq.ready && !initPending(i) && targetMatches(i))).asUInt.orR

  io.cmdResp <> responses.io.out

  // Ball bank ports occupy consecutive Channel slots in configuration order.
  channels.io.bankRead.zip(balls.flatMap(_.bankRead)).foreach { case (channel, port) => channel <> port }
  channels.io.bankWrite.zip(balls.flatMap(_.bankWrite)).foreach { case (channel, port) => channel <> port }
}

case class BallPorts(blink: BlinkIO, expansion: Option[Expansion])

trait BallFactory {

  def instantiate(
    fp: FrontendParam,
    mp: MemDomainParam,
    bp: BallDomainParam,
    id: Int
  ): BallPorts

}
