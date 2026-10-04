package framework.system.tile

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.top.GlobalConfig
import framework.memdomain.backend.MemRequestIO
import framework.memdomain.backend.shared.{SharedMemBackend, SharedMemLayout}
import framework.memdomain.backend.banks.btrace.PhysicalBankHash
import framework.memdomain.frontend.mem.MemConfigerIO
import framework.memdomain.isa.{MvoverCommand, MvoverISA, MvoverPort}
import memcore.memory.mesh_shm.MeshLocalBankPort

class BankClientPort(b: GlobalConfig) extends Bundle {
  val requests = Vec(SharedMemLayout.channelPerHart(b), Flipped(new MemRequestIO(b)))
  val move     = Flipped(new MvoverPort)

  val local = new MeshLocalBankPort(
    MvoverISA.AddressBits,
    MvoverISA.BankBits,
    b.memDomain.bankWidth,
    math.max(1, log2Ceil(b.frontend.rob_entries))
  )

  val config         = Flipped(Decoupled(new MemConfigerIO(b)))
  val queryValid     = Input(Bool())
  val queryVbank     = Input(UInt(b.memDomain.vbankIdWidth.W))
  val queryGroups    = Output(UInt(b.memDomain.groupCountWidth.W))
  val barrierArrive  = Input(Bool())
  val barrierRelease = Output(Bool())
}

/** Shared banks and inter-core bank movement; transport and CPU command decoding stay outside. */
@instantiable
class BankNetwork(
  b:              GlobalConfig,
  enabledCoreIds: Seq[Int],
  useMesh:        Boolean,
  controllerMove: Boolean)
    extends Module {
  require(b.memDomain.sharedEnable)
  require(enabledCoreIds.nonEmpty && enabledCoreIds == b.memDomain.computeCoreIds)
  require(enabledCoreIds.distinct.size == enabledCoreIds.size &&
    enabledCoreIds.forall(i => i >= 0 && i < b.memDomain.nCores))
  private val nCores          = b.memDomain.nCores
  private val channels        = SharedMemLayout.channelPerHart(b)
  private val controllerPorts = if (controllerMove) 1 else 0
  private val movePorts       = enabledCoreIds.size + controllerPorts

  @public
  val io = IO(new Bundle {
    val compute          = Vec(enabledCoreIds.size, new BankClientPort(b))
    val hartIds          = Input(Vec(nCores, UInt(b.tile.xLen.W)))
    val controllerMvover = if (controllerMove) Some(Flipped(new MvoverPort)) else None
    val bankHashes       =
      if (b.sim.diffTest) Some(Output(Vec(SharedMemLayout.totalBank(b), new PhysicalBankHash(b)))) else None
  })

  val sharedBackend = Instantiate(new SharedMemBackend(b, useMesh))
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
  val barrier = Instantiate(new BarrierUnit(enabledCoreIds.size))
  for (compute <- enabledCoreIds.indices) {
    barrier.io.arrive(compute)         := io.compute(compute).barrierArrive
    io.compute(compute).barrierRelease := barrier.io.release(compute)
  }
}
