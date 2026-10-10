package framework.system.tile

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.system.core.clink.ShmPort
import framework.memdomain.backend.shared.{SharedLeasePort, SharedMemBackend, SharedMemLayout, SharedPhysicalPort}
import framework.memdomain.backend.banks.btrace.PhysicalBankHash
import framework.memdomain.frontend.mem.MemConfigerIO
import framework.memdomain.isa.{MvoverCommand, MvoverPort}

/** Shared banks and inter-core bank movement; transport and CPU command decoding stay outside. */
@instantiable
class BankNetwork(
  b:              GlobalConfig,
  enabledCoreIds: Seq[Int],
  useMesh:        Boolean,
  controllerMove: Boolean,
  physicalPorts:  Int = 0)
    extends Module {
  require(b.memDomain.sharedEnable)
  require(enabledCoreIds.nonEmpty && enabledCoreIds == b.memDomain.computeCoreIds)
  require(enabledCoreIds.distinct.size == enabledCoreIds.size &&
    enabledCoreIds.forall(i => i >= 0 && i < b.memDomain.nCores))
  val nCores          = b.memDomain.nCores
  val channels        = SharedMemLayout.channelPerHart(b)
  val controllerPorts = if (controllerMove) 1 else 0
  val movePorts       = enabledCoreIds.size + controllerPorts

  @public
  val io = IO(new Bundle {
    val lease            = Option.when(physicalPorts > 0)(Flipped(new SharedLeasePort))
    val physical         = Vec(physicalPorts, Flipped(new SharedPhysicalPort(b.memDomain.bankWidth)))
    val compute          = Vec(enabledCoreIds.size, Flipped(new ShmPort(b)))
    val hartIds          = Input(Vec(nCores, UInt(b.tile.xLen.W)))
    val controllerMvover = if (controllerMove) Some(Flipped(new MvoverPort)) else None
    val bankHashes       =
      if (b.sim.diffTest) Some(Output(Vec(SharedMemLayout.totalBank(b), new PhysicalBankHash(b)))) else None
  })

  val sharedBackend: Instance[SharedMemBackend] = Instantiate(new SharedMemBackend(b, useMesh, physicalPorts))
  io.lease.foreach { lease =>
    sharedBackend.io.lease.get <> lease
    val endpoint = lease.request.bits.endpoint
    val owner    = MuxLookup(endpoint, 0.U(b.tile.xLen.W))(
      enabledCoreIds.zipWithIndex.map { case (physical, logical) => logical.U -> io.hartIds(physical) }
    )
    sharedBackend.io.leaseOwnerHartId.get := owner
    when(lease.request.valid && !reset.asBool) {
      assert(endpoint < enabledCoreIds.size.U, "Shared lease endpoint is not a configured NPU")
    }
  }
  sharedBackend.io.physical <> io.physical
  sharedBackend.io.physicalOwnerHartId.foreach(_ := io.hartIds(0))
  io.bankHashes.foreach(_ := sharedBackend.io.bank_hashes.get)
  for ((physical, compute) <- enabledCoreIds.zipWithIndex) {
    val port = io.compute(compute)
    for (channel <- 0 until channels) {
      sharedBackend.io.mem_req(compute * channels + channel) <> port.requests(channel)
    }
    sharedBackend.io.localBanks(physical) <> port.local
    sharedBackend.io.query_valid(physical)    := port.queryValid
    sharedBackend.io.query_hart_id(physical)  := io.hartIds(physical)
    sharedBackend.io.query_vbank_id(physical) := port.queryVbank
    port.queryGroups                          := sharedBackend.io.query_group_count(physical)
  }
  for (physical            <- 0 until nCores if !enabledCoreIds.contains(physical)) {
    val local = sharedBackend.io.localBanks(physical)
    local.request.ready                       := false.B
    local.response.valid                      := false.B
    local.response.bits                       := 0.U.asTypeOf(local.response.bits)
    sharedBackend.io.query_valid(physical)    := false.B
    sharedBackend.io.query_hart_id(physical)  := 0.U
    sharedBackend.io.query_vbank_id(physical) := 0.U
  }

  val moveArb         = Module(new Arbiter(new MvoverCommand, movePorts))
  val moveOwner       = RegInit(0.U(math.max(1, log2Ceil(movePorts)).W))
  val completionReady = Wire(Vec(movePorts, Bool()))
  io.controllerMvover.foreach { port =>
    moveArb.io.in(0) <> port.command
    port.completion.valid := sharedBackend.io.mvover.completion.valid && moveOwner === 0.U
    port.completion.bits  := sharedBackend.io.mvover.completion.bits
    completionReady(0)    := port.completion.ready
  }
  for (compute <- enabledCoreIds.indices) {
    val port = compute + controllerPorts
    moveArb.io.in(port) <> io.compute(compute).move.command
    io.compute(compute).move.completion.valid := sharedBackend.io.mvover.completion.valid && moveOwner === port.U
    io.compute(compute).move.completion.bits  := sharedBackend.io.mvover.completion.bits
    completionReady(port)                     := io.compute(compute).move.completion.ready
  }
  sharedBackend.io.mvover.command <> moveArb.io.out

  def physicalComputeCore(logical: UInt): UInt = MuxLookup(logical, nCores.U(8.W))(
    enabledCoreIds.zipWithIndex.map { case (physical, logicalIndex) => logicalIndex.U -> physical.U(8.W) }
  )

  sharedBackend.io.mvover.command.bits.sourceCore := physicalComputeCore(moveArb.io.out.bits.sourceCore)
  sharedBackend.io.mvover.command.bits.targetCore := physicalComputeCore(moveArb.io.out.bits.targetCore)
  sharedBackend.io.mvover.completion.ready        := completionReady(moveOwner)
  when(moveArb.io.out.fire)(moveOwner             := moveArb.io.chosen)

  val configArb = Module(new Arbiter(new MemConfigerIO(b), enabledCoreIds.size))
  for (compute <- enabledCoreIds.indices) { configArb.io.in(compute) <> io.compute(compute).config }
  sharedBackend.io.config <> configArb.io.out
  val barrier: Instance[BarrierUnit] = Instantiate(new BarrierUnit(enabledCoreIds.size))
  for (compute <- enabledCoreIds.indices) {
    barrier.io.arrive(compute)         := io.compute(compute).barrierArrive
    io.compute(compute).barrierRelease := barrier.io.release(compute)
  }
}
