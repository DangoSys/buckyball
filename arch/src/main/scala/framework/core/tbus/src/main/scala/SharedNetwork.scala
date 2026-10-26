package framework.core.tbus

import sims.configs.SimParam
import framework.lib.arbiter.{Arbiter}

import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.core.tbus.configs.{Params => TBusParam}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.core.tbus.configs.Layout
import framework.thread.tlink.{ExternalLink, SharedLink}
import framework.core.tbus.router.Router
import framework.acc.memdomain.{LayoutConfig, MvoverCommand, PhysicalBankHash}
import framework.ip.memory.mesh_shm.MeshSharedMemIO

/** Shared-bank interconnect: endpoint arbitration, mapping and synchronization. */
@instantiable
class SharedNetwork(
  val frontendParam: FrontendParam,
  val memParam:      MemDomainParam,
  val tbusParam:     TBusParam,
  val endpointIds:   Seq[Int],
  val traceIds:      Seq[Int],
  val simp:          SimParam)
    extends Module {
  val diffTest     = simp.diffTest
  val p            = tbusParam
  val participants = p.participants
  require(participants.nonEmpty && participants.distinct.size == participants.size)
  require(participants.forall(slot => slot >= 0 && slot < p.slots))
  val ports        = Layout.portsPerThread(memParam, tbusParam)

  @public
  val io = IO(new Bundle {
    val shm        = Flipped(new MeshSharedMemIO(p.router.mesh(frontendParam, memParam, tbusParam)))
    val threads    = Vec(participants.size, Flipped(new SharedLink(frontendParam, memParam)))
    val external   = Option.when(p.router.physicalPorts > 0 || p.externalMove)(new ExternalLink(memParam, tbusParam))
    val bankHashes = Option.when(diffTest)(Output(Vec(Layout.totalBank(memParam), new PhysicalBankHash(frontendParam))))
  })

  val router:  Instance[Router]  =
    Instantiate(new Router(frontendParam, memParam, tbusParam, endpointIds, traceIds, simp.diffTest))
  val barrier: Instance[Barrier] = Instantiate(new Barrier(tbusParam))

  router.io.memory <> io.shm
  io.bankHashes.foreach(_ := router.io.bank_hashes.get)

  val endpointSlots  = VecInit(participants.map(_.U(8.W)))
  val endpointOwners = VecInit(participants.map(slot => endpointIds(slot).U(64.W)))
  io.external.foreach { port =>
    router.io.physical <> port.memory
    port.lease.foreach { lease =>
      router.io.lease.get <> lease
      router.io.leaseOwnerThreadId.get := endpointOwners(lease.request.bits.endpoint)
      when(lease.request.valid && !reset.asBool) {
        assert(lease.request.bits.endpoint < participants.size.U, "Shared lease endpoint is outside TBus")
      }
    }
  }

  for ((slot, endpoint) <- participants.zipWithIndex) {
    val link = io.threads(endpoint)
    for (port <- 0 until ports) {
      router.io.mem_req(endpoint * ports + port) <> link.bank.requests(port)
    }
    router.io.localBanks(slot) <> link.bank.local
    router.io.mapping(slot) <> link.bank.mapping
    barrier.io.arrive(endpoint) := link.barrierArrive
    link.barrierRelease         := barrier.io.release(endpoint)
  }

  for (slot <- 0 until p.slots if !participants.contains(slot)) {
    val local = router.io.localBanks(slot)
    local.request.ready           := false.B
    local.response.valid          := false.B
    local.response.bits           := 0.U.asTypeOf(local.response.bits)
    router.io.mapping(slot).valid := false.B
    router.io.mapping(slot).bits  := 0.U.asTypeOf(router.io.mapping(slot).bits)
  }

  val moves     = io.external.toSeq.flatMap(_.move.toSeq) ++ io.threads.map(_.move)
  val moveArb   = Instantiate(new Arbiter(new MvoverCommand, moves.size))
  val moveOwner = Reg(UInt(math.max(1, log2Ceil(moves.size)).W))

  for ((port, endpoint) <- moves.zipWithIndex) {
    moveArb.io.in(endpoint) <> port.command
    port.completion.valid := router.io.mvover.completion.valid && moveOwner === endpoint.U
    port.completion.bits  := router.io.mvover.completion.bits
  }
  router.io.mvover.command <> moveArb.io.out
  router.io.mvover.command.bits.sourceThread := endpointSlots(moveArb.io.out.bits.sourceThread)
  router.io.mvover.command.bits.targetThread := endpointSlots(moveArb.io.out.bits.targetThread)
  router.io.mvover.completion.ready          := VecInit(moves.map(_.completion.ready))(moveOwner)
  when(moveArb.io.out.fire) {
    assert(
      moveArb.io.out.bits.sourceThread < participants.size.U &&
        moveArb.io.out.bits.targetThread < participants.size.U,
      "Bank movement endpoint is outside TBus"
    )
    moveOwner := moveArb.io.chosen
  }

  val layouts = Instantiate(new Arbiter(new LayoutConfig(frontendParam, memParam), participants.size))
  for (endpoint <- participants.indices) { layouts.io.in(endpoint) <> io.threads(endpoint).bank.config }
  router.io.config <> layouts.io.out
}

/** Core-local barrier; synchronization releases all participating endpoints together. */
@instantiable
class Barrier(val tbusParam: TBusParam) extends Module {
  val count = tbusParam.participants.size

  @public
  val io = IO(new Bundle {
    val arrive  = Input(Vec(count, Bool()))
    val release = Output(Vec(count, Bool()))
  })

  val arrived = RegInit(VecInit(Seq.fill(count)(false.B)))
  val release = arrived.asUInt.andR
  for (endpoint <- 0 until count) {
    when(io.arrive(endpoint))(arrived(endpoint) := true.B)
    io.release(endpoint)                        := release
  }
  when(release)(arrived.foreach(_ := false.B))
}
