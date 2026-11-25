package examples.hart.xiangshan

import framework.generic.acc.configs.BallLink
import framework.generic.contract.ball.BallHostParam

import framework.ball.bbus.BallFactory
import framework.ball.blink.BlinkIO

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import examples.hart.xiangshan.configs.Params
import framework.hart.cbus.HartFactory
import buckyball.config.{Design, HartPlacement}
import framework.hart.configs.HartParam
import framework.hart.hlink.{HLinkIO, HasHLink}
import framework.hart.hlink.configs.{Params => HLinkParam}
import framework.generic.bus.axi4
import framework.generic.bus.chi.{HomeMapping, RequestFlit}
import framework.generic.memory.interlock.MaintenanceOp
import framework.hart.memory.Requester
import framework.hart.memory.access.cache.Maintenance
import framework.generic.arbiter.RRArbiter

import framework.custom.frontend.Frontend
import framework.custom.memdomain.MemDomain
import framework.custom.gpdomain.GpDomain
import framework.ball.BallDomain
import framework.ball.blink.HasBlink
import framework.custom.debug.Debug
import framework.generic.contract.sims.configs.SimParam

@instantiable
class XiangShan(
  p:         Params,
  simp:      SimParam,
  factories: Map[String, BallFactory])
    extends Module
    with HasHLink {
  val diffTest = simp.diffTest

  @public
  val hlink = IO(new HLinkIO(XiangShan.hlink(p)))

  val native = Instantiate(new Native)
  native.io.clock := clock
  // The reset vector reaches the BPU through registers outside the reset; the BPU loads it in its first cycle out of
  // reset, so the core leaves reset 16 cycles after the hart, once the vector has arrived (else it fetches from 0).
  val resetCycles = RegInit(0.U(5.W))
  when(resetCycles =/= 16.U)(resetCycles := resetCycles + 1.U)
  native.io.reset                        := (reset.asBool || resetCycles =/= 16.U).asAsyncReset
  native.io.io_hartId                    := p.hartId.U
  native.io.io_nodeId                    := p.nodeId.U
  native.io.io_resetVector               := p.resetVector.U
  native.io.io_interrupts(0)             := hlink.interrupts(0).software
  native.io.io_interrupts(1)             := hlink.interrupts(0).timer
  native.io.io_interrupts(2)             := hlink.interrupts(0).external
  native.io.io_interrupts(3)             := hlink.interrupts(0).supervisorExternal

  val coherent = Instantiate(new Chi(p.chi, p.homeId))
  coherent.io.native <> native.io.io_chi

  val requester = Instantiate(new Requester(
    p.chi,
    p.nodeId,
    Seq(p.nodeId) ++ p.acc.toSeq.map(_ => p.nodeId),
    1
  ))

  requester.io.clients(0) <> coherent.io.requester
  hlink.memory.coherent(0) <> requester.io.coherent

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
  hlink.memory.uncached <> peripherals.io.out

  val time        = peripherals.io.in(1)
  val timePending = RegInit(false.B)
  time.aw.valid                  := false.B
  time.aw.bits                   := 0.U.asTypeOf(time.aw.bits)
  time.w.valid                   := false.B
  time.w.bits                    := 0.U.asTypeOf(time.w.bits)
  time.b.ready                   := true.B
  // mtime is read back to back, as a CLINT broadcasts time: every response
  // updates the core's time (and Sstc compares). A time CSR read is accepted
  // only as a new read issues, so the response it waits for is newer than it.
  time.ar.valid                  := !timePending
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
        simp,
        homeRanges = false
      ))
      val domain    =
        Instantiate(new BallDomain(
          BallLink.host(acc.frontend, acc.memDomain, acc.gpDomain),
          acc.ballDomain,
          p.permission,
          factories
        ))
      if (acc.gpDomain.enable) {
        val gp = Instantiate(new GpDomain(acc.frontend, acc.memDomain, acc.gpDomain))
        gp.io.ownerId := p.hartId.U
        gp.io.command <> frontend.io.gpRequest.get
        frontend.io.gpCompletion.get <> gp.io.complete
        gp.io.image <> memDomain.io.image.get
        memDomain.io.gpBanks.get <> gp.io.banks
        memDomain.io.gpLayoutRequest.get <> gp.io.layoutRequest
        gp.io.layoutResponse <> memDomain.io.gpLayoutResponse.get
        domain.io.operator.get.request <> gp.io.ball.request
        gp.io.ball.response <> domain.io.operator.get.response
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

      // The native hart drains older stores and stalls younger accesses during the command.
      memDomain.io.memory.cpuQuery := 0.U.asTypeOf(memDomain.io.memory.cpuQuery)
      val maintenance                                           = Instantiate(new Maintenance(
        p.chi,
        p.nodeId,
        HomeMapping(1, p.homeId),
        256,
        p.tracking,
        rangeHome = false
      ))
      val pteIdle :: pteCleaning :: pteRead :: pteReturn :: Nil = Enum(4)
      val pteState                                              = RegInit(pteIdle)
      val maintenancePte                                        = Reg(Bool())
      val cleanPte                                              = pteState === pteIdle && memDomain.io.memory.pteRequest.valid
      maintenance.io.request.valid          := cleanPte || memDomain.io.memory.maintenance.valid
      maintenance.io.request.bits           := memDomain.io.memory.maintenance.bits
      when(cleanPte) {
        maintenance.io.request.bits.tag       := 0.U
        maintenance.io.request.bits.firstLine := (memDomain.io.memory.pteRequest.bits.addr >> 6) << 6
        maintenance.io.request.bits.lastLine  := maintenance.io.request.bits.firstLine
        maintenance.io.request.bits.op        := MaintenanceOp.Clean.U
      }
      memDomain.io.memory.maintenance.ready := maintenance.io.request.ready && !cleanPte
      when(maintenance.io.request.fire) {
        maintenancePte          := cleanPte
        when(cleanPte)(pteState := pteCleaning)
      }
      memDomain.io.memory.maintained.valid  := maintenance.io.response.valid && !maintenancePte
      memDomain.io.memory.maintained.bits   := maintenance.io.response.bits
      maintenance.io.response.ready         := maintenancePte || memDomain.io.memory.maintained.ready
      when(maintenance.io.response.fire && maintenancePte) {
        assert(maintenance.io.response.bits.ok, "Page table maintenance failed")
        pteState := pteRead
      }
      maintenance.io.drained                := pending
      val requests            = Instantiate(new RRArbiter(
        new RequestFlit(p.chi),
        2
      ))
      requests.io.in(0) <> coherent.io.requester.req
      requests.io.in(1) <> maintenance.io.req
      requester.io.clients(0).req <> requests.io.out
      val response            = requester.io.clients(0).rxRsp
      val maintenanceResponse = response.bits.txnId === 256.U
      maintenance.io.rsp.valid          := response.valid && maintenanceResponse
      maintenance.io.rsp.bits           := response.bits
      coherent.io.requester.rxRsp.valid := response.valid && !maintenanceResponse
      response.ready                    := Mux(maintenanceResponse, maintenance.io.rsp.ready, coherent.io.requester.rxRsp.ready)

      val access   = Instantiate(new axi4.Interconnect(axi4.InterconnectParam(p.axi, 2, Some(16))))
      access.io.in(0) <> memDomain.io.ddr
      val pte      = access.io.in(1)
      val highWord = Reg(Bool())
      pte.aw.valid                               := false.B
      pte.aw.bits                                := 0.U.asTypeOf(pte.aw.bits)
      pte.w.valid                                := false.B
      pte.w.bits                                 := 0.U.asTypeOf(pte.w.bits)
      pte.b.ready                                := true.B
      pte.ar.valid                               := memDomain.io.memory.pteRequest.valid && pteState === pteRead
      pte.ar.bits                                := 0.U.asTypeOf(pte.ar.bits)
      pte.ar.bits.addr                           := memDomain.io.memory.pteRequest.bits.addr
      pte.ar.bits.size                           := 3.U
      pte.ar.bits.burst                          := 1.U
      memDomain.io.memory.pteRequest.ready       := pte.ar.ready && pteState === pteRead
      when(pte.ar.fire) { pteState := pteReturn; highWord := pte.ar.bits.addr(3) }
      memDomain.io.memory.pteResponse.valid      := pte.r.valid && pteState === pteReturn
      memDomain.io.memory.pteResponse.bits.data  := Mux(highWord, pte.r.bits.data(127, 64), pte.r.bits.data(63, 0))
      memDomain.io.memory.pteResponse.bits.error := pte.r.bits.resp =/= 0.U
      pte.r.ready                                := memDomain.io.memory.pteResponse.ready && pteState === pteReturn
      // No hart resolves DMA page faults here: a stalled instruction fails at once.
      memDomain.io.fault.resolve.valid           := memDomain.io.fault.fault.valid
      memDomain.io.fault.resolve.bits            := false.B
      when(pte.r.fire) { assert(pte.r.bits.last && pte.r.bits.id === 0.U); pteState := pteIdle }
      val dma = Instantiate(new DmaRequester(
        p.axi,
        p.chi,
        p.nodeId,
        HomeMapping(1, p.homeId)
      ))
      dma.io.axi <> access.io.out
      requester.io.clients(1) <> dma.io.coherent

      val debug = Instantiate(new Debug(acc.frontend, acc.memDomain, acc.gpDomain, acc.ballDomain, simp))
      // The accelerator serves its hart; its traces report as thread 0.
      frontend.io.ownerId                           := p.hartId.U
      memDomain.io.ownerId                          := p.hartId.U
      domain.io.ownerId                             := p.hartId.U
      debug.io.ids.trace                            := 0.U
      debug.io.ids.owner                            := p.hartId.U
      debug.io.ids.sharedOwner                      := 0.U
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

  hlink.memory.ddr.aw.valid := false.B
  hlink.memory.ddr.aw.bits  := 0.U.asTypeOf(hlink.memory.ddr.aw.bits)
  hlink.memory.ddr.w.valid  := false.B
  hlink.memory.ddr.w.bits   := 0.U.asTypeOf(hlink.memory.ddr.w.bits)
  hlink.memory.ddr.ar.valid := false.B
  hlink.memory.ddr.ar.bits  := 0.U.asTypeOf(hlink.memory.ddr.ar.bits)
  hlink.memory.ddr.b.ready  := false.B
  hlink.memory.ddr.r.ready  := false.B

}

object XiangShan {

  /** One hart whose native caches are one coherent CHI requester of an external Home. */
  def hlink(p: Params): HLinkParam = HLinkParam(1, false, Seq(p.nodeId), p.chi, p.axi, p.axi, false, false)
}

object XiangShanComponent extends HartFactory {
  val component = "xiangshan"

  def parameters(design: Design, source: HartPlacement): HartParam = {
    val native = Params.fromProto(source); require(native.acc.isEmpty)
    HartParam.fromProto(source, Seq.empty).copy(hlink = XiangShan.hlink(native))
  }

  def instantiate(
    design:   Design,
    source:   HartPlacement,
    hartp:    HartParam,
    simp:     SimParam,
    traceIds: Seq[Int]
  ): HLinkIO =
    Instantiate(new XiangShan(Params.fromProto(source), simp, Map.empty)).hlink

}
