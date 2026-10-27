package examples.core.xiangshan

import framework.acc.balldomain.bbus.BallFactory
import sims.configs.SimParam
import framework.acc.balldomain.ball.blink.BlinkIO

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import examples.core.xiangshan.configs.Params
import framework.core.clink.{CLinkIO, HasCLink}
import framework.core.clink.configs.{Params => CLinkParam}
import framework.ip.bus.axi4
import framework.ip.bus.chi.rnf.AxiCache

import framework.acc.frontend.Frontend
import framework.acc.memdomain.MemDomain
import framework.acc.gpdomain.GpDomain
import framework.acc.balldomain.BallDomain
import framework.acc.balldomain.ball.blink.HasBlink
import framework.acc.debug.Debug

@instantiable
class XiangShan(
  p:         Params,
  simp:      SimParam,
  factories: Map[String, BallFactory])
    extends Module
    with HasCLink {
  val diffTest = simp.diffTest

  @public
  val clink = IO(new CLinkIO(CLinkParam(
    1,
    false,
    Seq(p.nodeId) ++ p.acc.toSeq.map(_ => p.dmaCache.nodeId),
    p.chi,
    p.axi,
    p.axi,
    None
  )))

  val native = Instantiate(new Native)
  native.io.clock            := clock
  native.io.reset            := reset.asAsyncReset
  native.io.io_hartId        := p.hartId.U
  native.io.io_nodeId        := p.nodeId.U
  native.io.io_resetVector   := p.resetVector.U
  native.io.io_interrupts(0) := clink.interrupts(0).software
  native.io.io_interrupts(1) := clink.interrupts(0).timer
  native.io.io_interrupts(2) := clink.interrupts(0).external
  native.io.io_interrupts(3) := clink.interrupts(0).supervisorExternal

  val coherent = Instantiate(new Chi(p.chi, p.homeId))
  coherent.io.native <> native.io.io_chi
  clink.memory.coherent(0) <> coherent.io.requester

  val peripherals = Instantiate(new axi4.Interconnect(axi4.InterconnectParam(p.axi, 2, Some(16))))
  val upstream    = native.io.uncachedAXI(0)
  val downstream  = peripherals.io.in(0)
  downstream.aw.valid      := upstream.aw.valid
  upstream.aw.ready        := downstream.aw.ready
  downstream.aw.bits       := 0.U.asTypeOf(downstream.aw.bits)
  downstream.aw.bits.id    := upstream.aw.bits.id
  downstream.aw.bits.addr  := upstream.aw.bits.addr
  downstream.aw.bits.len   := upstream.aw.bits.len
  downstream.aw.bits.size  := upstream.aw.bits.size
  downstream.aw.bits.burst := upstream.aw.bits.burst
  downstream.aw.bits.lock  := upstream.aw.bits.lock
  downstream.aw.bits.cache := upstream.aw.bits.cache
  downstream.aw.bits.prot  := upstream.aw.bits.prot
  downstream.aw.bits.qos   := upstream.aw.bits.qos
  downstream.ar.valid      := upstream.ar.valid
  upstream.ar.ready        := downstream.ar.ready
  downstream.ar.bits       := 0.U.asTypeOf(downstream.ar.bits)
  downstream.ar.bits.id    := upstream.ar.bits.id
  downstream.ar.bits.addr  := upstream.ar.bits.addr
  downstream.ar.bits.len   := upstream.ar.bits.len
  downstream.ar.bits.size  := upstream.ar.bits.size
  downstream.ar.bits.burst := upstream.ar.bits.burst
  downstream.ar.bits.lock  := upstream.ar.bits.lock
  downstream.ar.bits.cache := upstream.ar.bits.cache
  downstream.ar.bits.prot  := upstream.ar.bits.prot
  downstream.ar.bits.qos   := upstream.ar.bits.qos
  downstream.w <> upstream.w
  upstream.b <> downstream.b
  upstream.r <> downstream.r
  clink.memory.uncached <> peripherals.io.out

  val time        = peripherals.io.in(1)
  val timePending = RegInit(false.B)
  time.aw.valid                  := false.B
  time.aw.bits                   := 0.U.asTypeOf(time.aw.bits)
  time.w.valid                   := false.B
  time.w.bits                    := 0.U.asTypeOf(time.w.bits)
  time.b.ready                   := true.B
  time.ar.valid                  := native.io.io_timeRequest.valid && !timePending
  time.ar.bits                   := 0.U.asTypeOf(time.ar.bits)
  time.ar.bits.addr              := p.mtimeAddress.U
  time.ar.bits.size              := 3.U
  time.ar.bits.burst             := 1.U
  native.io.io_timeRequest.ready := time.ar.ready && !timePending
  when(time.ar.fire)(timePending := true.B)
  time.r.ready                   := timePending
  native.io.io_time.valid        := time.r.fire
  native.io.io_time.bits         := (time.r.bits.data >> ((p.mtimeAddress & 15).toInt * 8))(63, 0)
  when(time.r.fire) {
    assert(time.r.bits.last && time.r.bits.resp === 0.U && time.r.bits.id === 0.U)
    timePending := false.B
  }

  p.acc match {
    case Some(acc) =>
      val frontend  =
        Instantiate(new Frontend(acc.frontend, acc.memDomain, acc.gpDomain, acc.ballDomain, p.permission, simp))
      val memDomain = Instantiate(new MemDomain(
        acc.frontend,
        acc.memDomain,
        acc.gpDomain,
        acc.transport,
        p.permission,
        p.tracking,
        p.walker,
        simp
      ))
      val domain    =
        Instantiate(new BallDomain(acc.frontend, acc.memDomain, acc.gpDomain, acc.ballDomain, p.permission, factories))
      if (acc.gpDomain.enable) {
        val gp = Instantiate(new GpDomain(acc.frontend, acc.memDomain, acc.gpDomain))
        gp.io.command <> frontend.io.gpRequest.get
        frontend.io.gpCompletion.get <> gp.io.complete
        gp.io.image <> memDomain.io.image.get
        memDomain.io.gpBanks.get <> gp.io.banks
        memDomain.io.gpLayoutRequest.get <> gp.io.layoutRequest
        gp.io.layoutResponse <> memDomain.io.gpLayoutResponse.get
        domain.io.kernel.get.request <> gp.io.ball.request
        gp.io.ball.response <> domain.io.kernel.get.response
      }
      val command   = native.io.io_commands.request
      frontend.io.rocc.cmd.valid                     := command.valid
      command.ready                                  := frontend.io.rocc.cmd.ready
      frontend.io.rocc.cmd.bits.instruction.raw_inst := command.bits.instruction
      frontend.io.rocc.cmd.bits.instruction.pc       := command.bits.pc
      frontend.io.rocc.cmd.bits.instruction.rs1Data  := command.bits.rs1
      frontend.io.rocc.cmd.bits.instruction.rs2Data  := command.bits.rs2
      frontend.io.rocc.cmd.bits.satp                 := command.bits.context.satp
      frontend.io.rocc.cmd.bits.effectivePrivilege   := command.bits.context.privilege
      frontend.io.rocc.cmd.bits.sum                  := command.bits.context.sum
      frontend.io.rocc.cmd.bits.mxr                  := command.bits.context.mxr
      for ((target, entry) <- frontend.io.rocc.cmd.bits.pmp.zip(command.bits.context.pmp)) {
        target.cfg  := entry.cfg.asTypeOf(target.cfg)
        target.addr := entry.addr
        val encoded = ((entry.addr.pad(48) << 1) | entry.cfg(3).asUInt | 1023.U)(47, 0)
        target.mask := (((encoded & ~(encoded + 1.U)) << 2) | 3.U)(47, 0)
      }
      frontend.io.rocc.resp.ready := true.B
      val pending = RegInit(false.B)
      when(command.fire)(pending                         := true.B)
      native.io.io_commands.complete.valid               := pending && (!frontend.io.rocc.busy || frontend.io.rocc.interrupt)
      native.io.io_commands.complete.bits                := frontend.io.rocc.interrupt
      when(native.io.io_commands.complete.valid)(pending := false.B)

      domain.io.issue <> frontend.io.ballRequest
      frontend.io.ballCompletion <> domain.io.complete
      memDomain.io.ball <> domain.io.memory
      memDomain.io.issue <> frontend.io.memoryRequest
      frontend.io.memoryCompletion <> memDomain.io.complete
      memDomain.io.mapping <> frontend.io.mappingUpdate
      frontend.io.barrier.release := frontend.io.barrier.arrive
      memDomain.io.banks <> domain.io.banks
      memDomain.io.ballLayoutRequest <> domain.io.layoutRequest
      domain.io.layoutResponse <> memDomain.io.ballLayoutResponse

      // DMA is a coherent requester alongside the native core. The submitting core has already drained stores
      // and cannot execute younger memory operations until this command completes.
      memDomain.io.memory.cpuQuery := 0.U.asTypeOf(memDomain.io.memory.cpuQuery)
      val maintained     = RegInit(false.B)
      val maintenanceTag = Reg(chiselTypeOf(memDomain.io.memory.maintenance.bits.tag))
      memDomain.io.memory.maintenance.ready                := !maintained
      when(memDomain.io.memory.maintenance.fire) {
        assert(pending)
        maintenanceTag := memDomain.io.memory.maintenance.bits.tag
        maintained     := true.B
      }
      memDomain.io.memory.maintained.valid                 := maintained
      memDomain.io.memory.maintained.bits.tag              := maintenanceTag
      memDomain.io.memory.maintained.bits.ok               := true.B
      when(memDomain.io.memory.maintained.fire)(maintained := false.B)

      val access     = Instantiate(new axi4.Interconnect(axi4.InterconnectParam(p.axi, 2, Some(16))))
      access.io.in(0) <> memDomain.io.ddr
      val pte        = access.io.in(1)
      val ptePending = RegInit(false.B)
      val highWord   = Reg(Bool())
      pte.aw.valid                               := false.B
      pte.aw.bits                                := 0.U.asTypeOf(pte.aw.bits)
      pte.w.valid                                := false.B
      pte.w.bits                                 := 0.U.asTypeOf(pte.w.bits)
      pte.b.ready                                := true.B
      pte.ar.valid                               := memDomain.io.memory.pteRequest.valid && !ptePending
      pte.ar.bits                                := 0.U.asTypeOf(pte.ar.bits)
      pte.ar.bits.addr                           := memDomain.io.memory.pteRequest.bits.addr
      pte.ar.bits.size                           := 3.U
      pte.ar.bits.burst                          := 1.U
      memDomain.io.memory.pteRequest.ready       := pte.ar.ready && !ptePending
      when(pte.ar.fire) { ptePending := true.B; highWord := pte.ar.bits.addr(3) }
      memDomain.io.memory.pteResponse.valid      := pte.r.valid && ptePending
      memDomain.io.memory.pteResponse.bits.data  := Mux(highWord, pte.r.bits.data(127, 64), pte.r.bits.data(63, 0))
      memDomain.io.memory.pteResponse.bits.error := pte.r.bits.resp =/= 0.U
      pte.r.ready                                := memDomain.io.memory.pteResponse.ready && ptePending
      when(pte.r.fire) { assert(pte.r.bits.last && pte.r.bits.id === 0.U); ptePending := false.B }
      val dma = Instantiate(new AxiCache(p.axi, p.dmaCache))
      dma.io.axi <> access.io.out
      dma.io.axi.ar.bits.cache                   := 15.U
      dma.io.axi.aw.bits.cache                   := 15.U
      clink.memory.coherent(1) <> dma.io.chi

      val debug = Instantiate(new Debug(acc.frontend, acc.memDomain, acc.gpDomain, acc.ballDomain, simp, 0, 0))
      debug.io.ant.valid                            := false.B
      debug.io.ant.bits                             := 0.U.asTypeOf(debug.io.ant.bits)
      debug.io.acc.frontend <> frontend.io.debug
      debug.io.acc.trace.memory                     := memDomain.io.traceMemory
      debug.io.acc.trace.ballStarted                := domain.io.traceStarted
      debug.io.acc.trace.ballCompleted              := domain.io.traceCompleted
      debug.io.acc.banks.zip(memDomain.io.bank_hashes).foreach { case (sink, src) => sink := src }
      for (port <- memDomain.io.shared.bank.requests) {
        port.read.req.ready   := false.B
        port.read.resp.valid  := false.B
        port.read.resp.bits   := 0.U.asTypeOf(port.read.resp.bits)
        port.write.req.ready  := false.B
        port.write.resp.valid := false.B
        port.write.resp.bits  := 0.U.asTypeOf(port.write.resp.bits)
      }
      memDomain.io.shared.bank.config.ready         := false.B
      memDomain.io.shared.bank.mapping.ready        := false.B
      memDomain.io.shared.bank.local.request.valid  := false.B
      memDomain.io.shared.bank.local.request.bits   := 0.U.asTypeOf(memDomain.io.shared.bank.local.request.bits)
      memDomain.io.shared.bank.local.response.ready := false.B
      memDomain.io.shared.move.command.ready        := false.B
      memDomain.io.shared.move.completion.valid     := false.B
      memDomain.io.shared.move.completion.bits      := false.B
    case None      =>
      native.io.io_commands.request.ready  := true.B
      native.io.io_commands.complete.valid := RegNext(native.io.io_commands.request.fire, false.B)
      native.io.io_commands.complete.bits  := true.B
  }
  clink.memory.ddr.aw.valid      := false.B
  clink.memory.ddr.aw.bits       := 0.U.asTypeOf(clink.memory.ddr.aw.bits)
  clink.memory.ddr.w.valid       := false.B
  clink.memory.ddr.w.bits        := 0.U.asTypeOf(clink.memory.ddr.w.bits)
  clink.memory.ddr.ar.valid      := false.B
  clink.memory.ddr.ar.bits       := 0.U.asTypeOf(clink.memory.ddr.ar.bits)
  clink.memory.ddr.b.ready       := true.B
  clink.memory.ddr.r.ready       := true.B
}
