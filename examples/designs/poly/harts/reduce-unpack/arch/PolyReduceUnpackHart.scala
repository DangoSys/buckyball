package examples.designs.poly.harts.reduceunpack

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
class PolyReduceUnpackHart(hartp: HartParam, simp: SimParam) extends Module with HasHLink {
  val diffTest = simp.diffTest
  val accp     = hartp.acc.get
  require(hartp.threads.isEmpty && !hartp.hlink.backing && hartp.hlink.requesters == 1)
  require(hartp.hlink.service && accp.memDomain.sharedEnable)
  require(hartp.tbus.get.router.physicalPorts == 0)
  @public
  val hlink    = IO(new HLinkIO(hartp.hlink))

  val loom:      Instance[PolyReduceUnpackLoom] = Instantiate(new PolyReduceUnpackLoom(hartp))
  val admission: Instance[DirectAdmission]      = Instantiate(new DirectAdmission(hartp.controller.commands))

  val acc: Instance[ReduceUnpackAcc] = Instantiate(new ReduceUnpackAcc(
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

  val payload = Instantiate(new ServiceTransfer(
    // The payload moves through a thread Transport: the hart accelerator's transport, as the thread's.
    framework.thread.acc.memdomain.tomem.configs.Params(
      accp.transport.tracking,
      accp.transport.prepared,
      accp.transport.axi,
      accp.transport.regions
    ),
    hartp.privileged.permission
  ))

  val transport = Instantiate(new framework.tlp.memory.DmaMemory(
    hartp.interconnect.bus,
    hartp.privileged.permission,
    hartp.memory,
    hartp.hlink.line.addressBits,
    hartp.hlink.backing,
    1
  ))

  loom.io.payload <> transport.io.services
  hlink.memory.page.get <> transport.io.page
  loom.io.payloadActive := payload.io.active
  val client = transport.io.clients(0)
  client.read <> payload.io.memory.read
  client.write <> payload.io.memory.write
  client.writeData <> payload.io.memory.writeData
  payload.io.memory.readData <> client.readData
  payload.io.memory.completed <> client.completed
  client.resolve.valid := client.fault.valid
  client.resolve.bits  := false.B
  transport.io.launch(0) <> payload.io.launch
  val command = admission.io.execution.command.bits
  payload.io.context.satp               := command.satp
  payload.io.context.effectivePrivilege := command.effectivePrivilege
  payload.io.context.sum                := command.sum
  payload.io.context.mxr                := command.mxr
  payload.io.context.pmp                := command.pmp
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

  val memoryBus = Instantiate(new framework.generic.bus.axi4.Interconnect(
    framework.generic.bus.axi4.InterconnectParam(hartp.hlink.ddr, 2, None)
  ))

  for ((client, index) <- Seq(acc.io.ddr, transport.io.axi(0)).zipWithIndex) {
    val widen = Instantiate(new framework.generic.bus.axi4.Widen(accp.transport.axi, hartp.hlink.ddr, 256))
    widen.io.in <> client
    memoryBus.io.in(index) <> widen.io.out
  }
  hlink.memory.ddr <> memoryBus.io.out

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
  barrier.io.arrive(0)      := acc.io.shm.barrierArrive
  acc.io.shm.barrierRelease := barrier.io.release(0)
  acc.io.shared_bank_hashes.zip(banks.io.bankHashes).foreach { case (sink, source) => sink := source }

  payload.io.host <> admission.io.remote
  hlink.service.get <> payload.io.remote
  hlink.payload.get <> payload.io.payload

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

object PolyReduceUnpack extends HartFactory {
  val component = "reduce-unpack"
  def parameters(design: Design, source: HartPlacement): HartParam =
    HartParam.fromProto(design, source, externalHome = true)

  def instantiate(
    design:   Design,
    source:   HartPlacement,
    hartp:    HartParam,
    simp:     SimParam,
    traceIds: Seq[Int]
  ): HLinkIO =
    Instantiate(new PolyReduceUnpackHart(hartp, simp)).hlink

}
