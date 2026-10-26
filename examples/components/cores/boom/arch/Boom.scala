package examples.core.boom

import framework.acc.balldomain.bbus.BallFactory
import sims.configs.SimParam
import framework.acc.balldomain.ball.blink.BlinkIO

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import examples.core.boom.configs.Params
import framework.core.clink.{CLinkIO, HasCLink}
import framework.core.clink.configs.{Params => CLinkParam}
import framework.ip.bus.axi4

import framework.acc.frontend.Frontend
import framework.acc.memdomain.MemDomain
import framework.acc.gpdomain.GpDomain
import framework.acc.balldomain.BallDomain
import framework.acc.balldomain.ball.blink.HasBlink
import framework.acc.debug.Debug

@instantiable
class Boom(
  p:         Params,
  simp:      SimParam,
  factories: Map[String, BallFactory])
    extends Module
    with HasCLink {
  val diffTest = simp.diffTest
  @public
  val clink    = IO(new CLinkIO(CLinkParam(1, false, Seq.empty, p.chi, p.axi, p.axi, None)))
  val native   = Instantiate(new Native)
  native.io.clock            := clock
  native.io.reset            := reset.asBool
  native.io.io_resetVector   := p.resetVector.U
  native.io.io_interrupts(0) := clink.interrupts(0).software
  native.io.io_interrupts(1) := clink.interrupts(0).timer
  native.io.io_interrupts(2) := clink.interrupts(0).external
  native.io.io_interrupts(3) := clink.interrupts(0).supervisorExternal
  for (
    (up, down) <- Seq((native.io.memoryAXI(0), clink.memory.ddr), (native.io.peripheralsAXI(0), clink.memory.uncached))
  ) {
    down.aw.valid := up.aw.valid
    up.aw.ready   := down.aw.ready
    down.ar.valid := up.ar.valid
    up.ar.ready   := down.ar.ready
    for ((dst, src) <- Seq((down.aw.bits, up.aw.bits), (down.ar.bits, up.ar.bits))) {
      dst       := 0.U.asTypeOf(dst)
      dst.id    := src.id; dst.addr    := src.addr; dst.len   := src.len; dst.size   := src.size
      dst.burst := src.burst; dst.lock := src.lock; dst.cache := src.cache; dst.prot := src.prot; dst.qos := src.qos
    }
    down.w <> up.w
    up.b <> down.b
    up.r <> down.r
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
      frontend.io.rocc.cmd.bits.instruction.pc       := 0.U
      frontend.io.rocc.cmd.bits.instruction.rs1Data  := command.bits.rs1
      frontend.io.rocc.cmd.bits.instruction.rs2Data  := command.bits.rs2
      frontend.io.rocc.cmd.bits.satp                 := native.io.io_context.satp
      frontend.io.rocc.cmd.bits.effectivePrivilege   := command.bits.privilege
      frontend.io.rocc.cmd.bits.sum                  := command.bits.sum
      frontend.io.rocc.cmd.bits.mxr                  := command.bits.mxr
      for ((target, entry) <- frontend.io.rocc.cmd.bits.pmp.zip(native.io.io_context.pmp)) {
        target.cfg  := entry.cfg.asTypeOf(target.cfg)
        target.addr := entry.addr
        val encoded = ((entry.addr.pad(44) << 1) | entry.cfg(3).asUInt)(43, 0)
        target.mask := (((encoded & ~(encoded + 1.U)) << 2) | 3.U)(43, 0)
      }
      frontend.io.rocc.resp.ready := true.B
      val pending = frontend.io.rocc.busy
      native.io.io_commands.busy  := frontend.io.rocc.busy || command.valid
      native.io.io_commands.fault := frontend.io.rocc.interrupt

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

      // The native coherence hub snoops BOOM caches for DMA. The submitting core has already drained stores
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
      val dma = native.io.dmaAXI(0)
      val upstream = access.io.out
      dma.aw.valid      := upstream.aw.valid
      upstream.aw.ready := dma.aw.ready
      dma.ar.valid      := upstream.ar.valid
      upstream.ar.ready := dma.ar.ready
      for ((dst, src) <- Seq((dma.aw.bits, upstream.aw.bits), (dma.ar.bits, upstream.ar.bits))) {
        dst.id    := src.id; dst.addr    := src.addr; dst.len   := src.len; dst.size := src.size
        dst.burst := src.burst; dst.lock := src.lock; dst.cache := 15.U; dst.prot    := src.prot; dst.qos := src.qos
      }
      dma.w <> upstream.w
      upstream.b <> dma.b
      upstream.r <> dma.r

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
      native.io.io_commands.request.ready := true.B
      native.io.io_commands.busy          := false.B
      native.io.io_commands.fault         := RegNext(native.io.io_commands.request.fire, false.B)
      val dma = native.io.dmaAXI(0)
      dma.aw.valid := false.B; dma.aw.bits := 0.U.asTypeOf(dma.aw.bits)
      dma.w.valid  := false.B; dma.w.bits  := 0.U.asTypeOf(dma.w.bits)
      dma.ar.valid := false.B; dma.ar.bits := 0.U.asTypeOf(dma.ar.bits)
      dma.b.ready  := true.B; dma.r.ready  := true.B
  }
}
