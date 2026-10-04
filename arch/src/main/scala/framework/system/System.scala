package framework.system

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.system.configloader.{ExampleTopology, RocketTileCore}
import framework.system.tile.Tile
import framework.system.memory.Memory
import framework.memdomain.frontend.mem.dma.DmaStatus
import framework.system.device.{BootRomLines, DeviceParams, Devices}
import memcore.bus.chi.rnf.RnfParams
import memcore.memory.coherence.configs.CoherenceParams
import memcore.memory.cpu.{CpuMemParams, PhysicalRegion, UncachedRequest, UncachedResponse}
import memcore.memory.interlock.{Params => TrackingParams}
import memcore.memory.uncached_ram.{Params => RamParams}
import memcore.memory.ddr.{Params => DdrParams}

/**
 * The chip: the main tile and its compute tiles, chip CLINT/PLIC and the DDR boundary. Only DDR
 * AXI, the remaining device peers and the PLIC interrupt sources are external.
 *
 * Per-core ports are concatenated in tile order (tile 0 first), which is also the order of
 * `ram.cpuHartIds`. Each tile keeps its own L2 and CHI fabric; tiles share only the chip memory.
 * `memory` is a per-tile template whose agent count each tile derives from its core count.
 */
@instantiable
class System(
  topology:        ExampleTopology,
  cpuPhysicalBits: Int,
  memory:          CoherenceParams,
  l1:              RnfParams,
  regions:         Seq[PhysicalRegion],
  tracking:        TrackingParams,
  ram:             RamParams,
  ddr:             DdrParams,
  devices:         DeviceParams = DeviceParams())
    extends Module {
  private val tiles   = topology.tiles
  private val hartIds = tiles.flatMap(_.hartIds)
  require(ram.cpuHartIds == hartIds.map(BigInt(_)) && ram.lineAgents == tiles.size)
  require(ram.line == memory.chi)
  private val n       = hartIds.size

  private val dmaCounts = tiles.map(_.cores.count {
    case RocketTileCore(_, Some(_)) => true
    case _                          => false
  })

  private val dmaMasters = dmaCounts.sum

  private val cp = CpuMemParams(memory.chi, ram.tagBits)

  @public
  val io = IO(new Bundle {
    val resetVector       = Input(Vec(n, UInt(64.W)))
    val interruptSources  = Input(UInt(devices.plic.sources.W))
    val deviceRequest     = Vec(n, Decoupled(new UncachedRequest(cp)))
    val deviceResponse    = Vec(n, Flipped(Decoupled(new UncachedResponse(cp))))
    val axi               = new memcore.bus.axi4.Port(ddr.axi)
    val failure           = Output(Vec(n, Valid(new DmaStatus)))
    val retired           = Output(Vec(n, Bool()))
    val retiredPc         = Output(Vec(n, UInt(64.W)))
    val trapped           = Output(Vec(n, Bool()))
    val trapCause         = Output(Vec(n, UInt(64.W)))
    val trapValue         = Output(Vec(n, UInt(64.W)))
    val trapPc            = Output(Vec(n, UInt(64.W)))
    val workDrained       = Output(Vec(n, Bool()))
    val memoryOutstanding = Output(UInt(log2Ceil(ram.slots + 2 * dmaMasters + 1).W))
  })

  val tileInstances = tiles.map { t =>
    Instantiate(new Tile(
      t,
      Tile.cpuParameters(t, hartIds.max, cpuPhysicalBits),
      memory.copy(agents = 2 * t.cores.size),
      l1,
      regions,
      tracking,
      axiParams = ddr.axi
    ))
  }

  val backing     = Instantiate(new Memory(ram, ddr, dmaMasters))
  val chipDevices = Module(new Devices(devices, cp, hartIds))
  chipDevices.io.sources := io.interruptSources
  chipDevices.io.request <> backing.io.deviceRequest
  backing.io.deviceResponse <> chipDevices.io.response
  io.deviceRequest <> chipDevices.io.externalRequest
  chipDevices.io.externalResponse <> io.deviceResponse
  io.axi <> backing.io.axi
  io.memoryOutstanding   := backing.io.outstanding

  private val coreBase = tiles.scanLeft(0)(_ + _.cores.size)
  private val dmaBase  = dmaCounts.scanLeft(0)(_ + _)
  for ((tile, index) <- tileInstances.zipWithIndex) {
    for (local    <- tiles(index).cores.indices) {
      val core = coreBase(index) + local
      tile.io.resetVector(local) := io.resetVector(core)
      tile.io.interrupts(local)  := chipDevices.io.interrupts(core)
      backing.io.cpuRequest(core) <> tile.io.uncachedRequest(local)
      tile.io.uncachedResponse(local) <> backing.io.cpuResponse(core)
      io.failure(core)           := tile.io.failure(local)
      io.retired(core)           := tile.io.retired(local)
      io.retiredPc(core)         := tile.io.retiredPc(local)
      io.trapped(core)           := tile.io.trapped(local)
      io.trapCause(core)         := tile.io.trapCause(local)
      io.trapValue(core)         := tile.io.trapValue(local)
      io.trapPc(core)            := tile.io.trapPc(local)
      io.workDrained(core)       := tile.io.workDrained(local)
    }
    devices.bootrom match {
      case Some(rom) =>
        val lines = Module(new BootRomLines(rom, memory.chi))
        lines.io.request <> tile.io.backingRequest(0)
        tile.io.backingResponse(0) <> lines.io.response
        backing.io.lineRequest(index) <> lines.io.memoryRequest
        lines.io.memoryResponse <> backing.io.lineResponse(index)
      case None      =>
        backing.io.lineRequest(index) <> tile.io.backingRequest(0)
        tile.io.backingResponse(0) <> backing.io.lineResponse(index)
    }
    for (endpoint <- 0 until dmaCounts(index)) {
      backing.io.dma(dmaBase(index) + endpoint) <> tile.io.dma(endpoint)
    }
  }
}
