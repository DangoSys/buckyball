package framework.acc.balldomain

import framework.lib.arbiter.{Arbiter}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.gpdomain.configs.GpDomainParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.core.memory.access.preflight.configs.PmpParams
import framework.acc.balldomain.ball.blink.BallResponse
import framework.acc.balldomain.bbus.{BBus, BallFactory, BallRequest}
import framework.acc.balldomain.decoder.Decoder
import framework.acc.balldomain.sub_rob.{SubROB, SubRobRow}
import framework.acc.balldomain.mmio_bank.{MetadataWrite, MmioBank}
import framework.acc.frontend.scheduler.{Complete, Issue, MemoryIssue}
import framework.acc.memdomain.{BankLayout, BankLayoutRequest, BankPort, Response => MemoryResponse}

/** GP kernel Ball calls; the GP domain resolves its launch groups before issuing. */
class KernelBallPort(f: FrontendParam, m: MemDomainParam) extends Bundle {
  val request  = Decoupled(new BallRequest(f, m))
  val response = Flipped(Decoupled(new BallResponse(f)))
}

/** BallDomain owns child identities; MemDomain receives ordinary tagged memory operations. */
class DomainMemory(f: FrontendParam, m: MemDomainParam, p: PmpParams) extends Bundle {
  val request  = Decoupled(new MemoryIssue(f, m, p))
  val response = Flipped(Decoupled(new MemoryResponse(f)))
  val metadata = new MetadataWrite(m)
}

@instantiable
class BallDomain(
  fp:        FrontendParam,
  mp:        MemDomainParam,
  gp:        GpDomainParam,
  bp:        BallDomainParam,
  pp:        PmpParams,
  factories: Map[String, BallFactory])
    extends Module {

  @public
  val io = IO(new Bundle {
    val issue          = Flipped(Decoupled(new MemoryIssue(fp, mp, pp)))
    val complete       = Decoupled(new MemoryResponse(fp))
    val memory         = new DomainMemory(fp, mp, pp)
    val banks          = Vec(mp.bankRequestPorts, new BankPort(fp, mp))
    val layoutRequest  = Decoupled(new BankLayoutRequest(fp))
    val layoutResponse = Flipped(Decoupled(new BankLayout(mp)))
    val kernel         = Option.when(gp.enable)(Flipped(new KernelBallPort(fp, mp)))
    val traceStarted   = Output(Vec(bp.balls.size, Valid(UInt(log2Ceil(fp.rob_entries).W))))
    val traceCompleted = Output(Vec(bp.balls.size, Valid(UInt(log2Ceil(fp.rob_entries).W))))
  })

  val bbus: Instance[BBus] = Instantiate(new BBus(fp, mp, gp, bp, factories))
  val expander = bbus.expansion
  if (expander.nonEmpty) require(
    mp.mmioEnable && !fp.bank_alias_enable,
    "Expanded instructions own architectural scratch banks until their parent completes"
  )
  val decoder: Instance[Decoder] = Instantiate(new Decoder(fp, mp, bp))
  bbus.io.cmdReq <> decoder.io.command

  // Frontend and SubROB work enter as an Issue; the Ball sees only its BLink command.
  val operation = Wire(new Issue(fp, mp))
  val access    = operation.renamedBankAccess
  val request   = decoder.io.issue.bits
  request.target           := 0.U
  request.instId           := operation.instId
  request.enabled(0)       := access.rd_bank_0_valid
  request.enabled(1)       := access.rd_bank_1_valid
  request.enabled(2)       := access.wr_bank_valid
  request.cmd.rob_id       := operation.rob_id
  request.cmd.funct        := operation.cmd.cmd.raw_inst(31, 25)
  request.cmd.rs2          := operation.cmd.cmd.rs2Data
  request.cmd.iter         := operation.cmd.iter
  request.cmd.op1_bank     := access.rd_bank_0_id
  request.cmd.op2_bank     := access.rd_bank_1_id
  request.cmd.wr_bank      := access.wr_bank_id
  request.cmd.op1_group    := 0.U
  request.cmd.op2_group    := 0.U
  request.cmd.wr_group     := 0.U
  request.cmd.op1_col      := operation.cmd.op1_col
  request.cmd.op2_col      := operation.cmd.op2_col
  request.cmd.wr_col       := operation.cmd.wr_col
  request.cmd.wr_read_bank := access.rd_wr_id
  request.cmd.wr_arch_bank := operation.cmd.architecturalDestination
  io.banks <> bbus.io.banks
  io.layoutRequest <> bbus.io.layoutRequest
  bbus.io.layoutResponse <> io.layoutResponse
  io.traceStarted          := bbus.io.traceStarted
  io.traceCompleted        := bbus.io.traceCompleted
  if (gp.enable) {
    val kernelDecoder: Instance[Decoder] = Instantiate(new Decoder(fp, mp, bp))
    kernelDecoder.io.issue <> io.kernel.get.request
    bbus.io.kernelReq.get <> kernelDecoder.io.command
    io.kernel.get.response <> bbus.io.kernelResp.get
  }
  if (mp.mmioEnable) {
    val reads  = bp.balls.map(_.mmioReadBW).sum
    val writes = bp.balls.map(_.mmioWriteBW).sum
    val mmio: Instance[MmioBank] = Instantiate(new MmioBank(mp, reads + expander.size, writes + expander.size))
    mmio.io.upload <> io.memory.metadata
    for (i <- 0 until reads) mmio.io.reads(i) <> bbus.io.mmioRead(i)
    for (i <- 0 until writes) mmio.io.writes(i) <> bbus.io.mmioWrite(i)
    expander.foreach { e => mmio.io.reads(reads) <> e.read; mmio.io.writes(writes) <> e.write }
  } else {
    require(bp.balls.forall(b => b.mmioReadBW == 0 && b.mmioWriteBW == 0))
    io.memory.metadata.request.ready  := false.B
    io.memory.metadata.response.valid := false.B
    io.memory.metadata.response.bits  := false.B
    when(io.memory.metadata.request.valid)(assert(false.B, "MMIO metadata bank is disabled"))
  }

  io.complete.bits         := 0.U.asTypeOf(io.complete.bits)
  io.memory.request.valid  := false.B
  io.memory.request.bits   := 0.U.asTypeOf(io.memory.request.bits)
  io.memory.response.ready := false.B
  expander match {
    case None    =>
      decoder.io.issue.valid  := io.issue.valid
      operation               := io.issue.bits.operation
      io.issue.ready          := decoder.io.issue.ready
      io.complete.valid       := bbus.io.cmdResp.valid
      io.complete.bits.rob_id := bbus.io.cmdResp.bits.rob_id
      bbus.io.cmdResp.ready   := io.complete.ready
    case Some(e) =>
      val sub: Instance[SubROB] = Instantiate(new SubROB(fp, mp, bp))
      val active        = RegInit(false.B)
      val parent        = Reg(chiselTypeOf(io.issue.bits))
      val ballPending   = RegInit(false.B)
      val memoryPending = RegInit(false.B)
      val ballTag       = Reg(chiselTypeOf(sub.io.issueSubId(0)))
      val memoryTag     = Reg(chiselTypeOf(sub.io.issueSubId(1)))
      val faulted       = RegInit(false.B)
      val faultReported = RegInit(false.B)
      val fault         = Reg(chiselTypeOf(io.complete.bits.fault))
      e.command.bits.rob_id := io.issue.bits.operation.rob_id
      e.command.bits.funct  := io.issue.bits.operation.cmd.cmd.raw_inst(31, 25)
      e.command.bits.rs2    := io.issue.bits.operation.cmd.cmd.rs2Data
      io.issue.ready        := !active && !faulted && Mux(e.accepts, e.command.ready, decoder.io.issue.ready)
      e.command.valid       := io.issue.valid && e.accepts && !active && !faulted
      when(io.issue.fire && e.starts) { active := true.B; parent := io.issue.bits }
      sub.io.write <> e.rows
      sub.io.write.valid    := e.rows.valid && active && !faulted
      e.rows.ready          := sub.io.write.ready && active && !faulted
      val child = Wire(new Issue(fp, mp))
      child                                              := parent.operation
      child.cmd                                          := sub.io.issue(0).bits
      child.cmd.cmd.pc                                   := parent.operation.cmd.cmd.pc
      child.renamedBankAccess                            := sub.io.issue(0).bits.bankAccess
      decoder.io.issue.valid                             := !faulted && Mux(
        active,
        sub.io.issue(0).valid && !ballPending,
        io.issue.valid && !e.accepts
      )
      operation                                          := Mux(active, child, io.issue.bits.operation)
      sub.io.issue(0).ready                              := active && !faulted && !ballPending && decoder.io.issue.ready
      when(sub.io.issue(0).fire) { ballPending := true.B; ballTag := sub.io.issueSubId(0) }
      io.memory.request.valid                            := active && !faulted && sub.io.issue(1).valid && !memoryPending
      io.memory.request.bits                             := parent
      io.memory.request.bits.operation.cmd               := sub.io.issue(1).bits
      io.memory.request.bits.operation.cmd.cmd.pc        := parent.operation.cmd.cmd.pc
      io.memory.request.bits.operation.renamedBankAccess := sub.io.issue(1).bits.bankAccess
      sub.io.issue(1).ready                              := active && !faulted && !memoryPending && io.memory.request.ready
      when(io.memory.request.fire) { memoryPending := true.B; memoryTag := sub.io.issueSubId(1) }
      val completed = Instantiate(new Arbiter(chiselTypeOf(ballTag), 2))
      completed.io.in(0).valid                           := active && ballPending && bbus.io.cmdResp.valid
      completed.io.in(0).bits                            := ballTag
      completed.io.in(1).valid                           := active && memoryPending && io.memory.response.valid
      completed.io.in(1).bits                            := memoryTag
      sub.io.subComplete <> completed.io.out
      io.memory.response.ready                           := memoryPending && completed.io.in(1).ready
      when(completed.io.in(0).fire)(ballPending          := false.B)
      when(io.memory.response.fire) {
        memoryPending := false.B
        when(io.memory.response.bits.fault.error =/= 0.U) {
          faulted := true.B; fault := io.memory.response.bits.fault
        }
      }
      val responses = Instantiate(new Arbiter(new Complete(fp), 3))
      responses.io.in(0).valid                           := !active && bbus.io.cmdResp.valid
      responses.io.in(0).bits.rob_id                     := bbus.io.cmdResp.bits.rob_id
      responses.io.in(1) <> e.complete
      responses.io.in(2).valid                           := sub.io.masterComplete.valid && !faulted
      responses.io.in(2).bits.rob_id                     := sub.io.masterComplete.bits
      sub.io.masterComplete.ready                        := responses.io.in(2).ready && !faulted
      when(sub.io.masterComplete.fire)(active            := false.B)
      bbus.io.cmdResp.ready                              := Mux(active, ballPending && completed.io.in(0).ready, responses.io.in(0).ready)
      io.complete.valid                                  := Mux(faulted, !faultReported, responses.io.out.valid)
      io.complete.bits.rob_id                            := Mux(faulted, parent.operation.rob_id, responses.io.out.bits.rob_id)
      io.complete.bits.fault                             := Mux(faulted, fault, 0.U.asTypeOf(fault))
      responses.io.out.ready                             := io.complete.ready && !faulted
      when(io.complete.fire && faulted)(faultReported    := true.B)
  }
}
