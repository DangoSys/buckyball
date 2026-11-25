package examples.hart.reduce

import framework.hart.cbus.HartFactory
import buckyball.config.{Design, HartPlacement}

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.hart.configs.HartParam
import framework.hart.hlink.{HLinkIO, HasHLink}
import framework.tlp.controller.admission.DirectAdmission
import framework.tlp.tbus.{Barrier, SharedNetwork}
import framework.tlp.tbus.configs.Layout
import framework.tlp.shm.SHM
import framework.custom.debug.Debug
import framework.hart.memory.{Requester}
import framework.generic.contract.sims.configs.SimParam

/** Native execution hart with a direct BBus accelerator and shared-bank C2C access. */
@instantiable
class ReduceHart(hartp: HartParam, simp: SimParam) extends Module with HasHLink {
  val diffTest = simp.diffTest
  val accp     = hartp.acc.get
  require(hartp.threads.isEmpty && !hartp.hlink.backing && hartp.hlink.requesters == 1)
  require(hartp.hlink.service && accp.memDomain.sharedEnable)
  require(hartp.tbus.get.router.physicalPorts == 0)
  @public
  val hlink    = IO(new HLinkIO(hartp.hlink))

  val loom:      Instance[Loom]            = Instantiate(new Loom(hartp))
  val admission: Instance[DirectAdmission] = Instantiate(new DirectAdmission(hartp.controller.commands))

  val acc: Instance[Acc] = Instantiate(new Acc(
    accp,
    hartp.privileged.permission,
    hartp.memory.dcache.tracking,
    hartp.memory.memoryAccess.walker,
    simp,
    homeRanges = hartp.hlink.pages
  ))

  loom.io.interrupts := hlink.interrupts(0)
  admission.io.execution <> loom.io.commands
  loom.io.drained    := admission.io.drained
  admission.io.accelerator <> acc.io.rocc
  loom.io.memory <> acc.io.memory
  loom.io.fault <> acc.io.fault
  hlink.memory.uncached <> loom.io.uncached

  val requester = Instantiate(new Requester(
    hartp.hlink.line,
    hartp.hlink.requesterIds.head,
    Seq(hartp.memory.icache.nodeId, hartp.memory.dcache.cache.nodeId),
    2
  ))

  requester.io.clients.take(2).zip(loom.io.coherent).foreach { case (port, cache) => port <> cache }
  hlink.memory.coherent(0) <> requester.io.coherent

  // The accelerator's transport rows are narrower than the hart's DDR port:
  // widen them, as a connection between the two widths would truncate data.
  val widen = Instantiate(new framework.generic.bus.axi4.Widen(accp.transport.axi, hartp.hlink.ddr, 256))
  widen.io.in <> acc.io.ddr
  hlink.memory.ddr <> widen.io.out
  // No thread DMA translates through the Home's pages: none asks.
  hlink.memory.page.foreach(_.address := 0.U)

  val banks =
    Instantiate(new SharedNetwork(
      accp.frontend,
      accp.memDomain,
      hartp.tbus.get,
      Seq(hartp.hartId),
      Seq(hartp.hartId),
      simp,
      0
    ))

  val shm     = Instantiate(new SHM(accp.frontend, accp.memDomain, hartp.tbus.get))
  val barrier = Instantiate(new Barrier(hartp.tbus.get))
  shm.io <> banks.io.shm
  banks.io.banks(0) <> acc.io.shm.bank
  banks.io.moves(0) <> acc.io.shm.move
  barrier.io.arrive(0)            := acc.io.shm.barrierArrive
  acc.io.shm.barrierRelease       := barrier.io.release(0)
  acc.io.shared_bank_hashes.zip(banks.io.bankHashes).foreach { case (sink, source) => sink := source }

  hlink.service.get <> admission.io.remote
  hlink.payload.get.send.valid    := false.B
  hlink.payload.get.send.bits     := 0.U
  hlink.payload.get.receive.ready := false.B

  val debug = Instantiate(new Debug(
    accp.frontend,
    accp.memDomain,
    accp.gpDomain,
    accp.ballDomain,
    simp
  ))

  // The accelerator serves its hart.
  acc.io.ownerId           := hartp.hartId.U
  debug.io.ids.trace       := hartp.hartId.U
  debug.io.ids.owner       := hartp.hartId.U
  debug.io.ids.sharedOwner := hartp.hartId.U

  debug.io.ant.valid := false.B
  debug.io.ant.bits  := 0.U.asTypeOf(debug.io.ant.bits)
  debug.io.acc <> acc.io.debug
}

object Reduce extends HartFactory {
  val component = "reduce"
  def parameters(design: Design, source: HartPlacement): HartParam =
    HartParam.fromProto(design, source, externalHome = true)

  def instantiate(
    design:   Design,
    source:   HartPlacement,
    hartp:    HartParam,
    simp:     SimParam,
    traceIds: Seq[Int]
  ): HLinkIO =
    Instantiate(new ReduceHart(hartp, simp)).hlink

}
