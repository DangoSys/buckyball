package sims.soc

import chisel3._
import chisel3.experimental.hierarchy.Instance
import framework.system.{System, SystemParams}
import framework.system.configloader.{AntTileCore, ChipLoader, RocketTileCore}
import framework.system.device.{BootRom, DeviceParams}
import memcore.bus.axi4
import memcore.bus.chi.{Params => ChiParams}
import memcore.bus.chi.rnf.RnfParams
import memcore.memory.cache.configs.CacheParams
import memcore.memory.coherence.configs.CoherenceParams
import memcore.memory.cpu.{CpuMemParams, PhysicalRegion}
import memcore.memory.ddr.{Params => DdrParams}
import memcore.memory.interlock.{Params => TrackingParams}
import sims.scu.{SCUParams}

/** A chip on the explicit System: one PB plus the platform's DRAM window. */
abstract class SystemTarget(val pb: String) {
  def instantiate(p: SystemParams): Instance[System]
  val dramBase:  BigInt = BigInt("80000000", 16)
  val dramBytes: BigInt = BigInt(512) << 20
}

/**
 * The explicit System with its SCU and NPU-fault abort. Every platform (Verilator, P2E) supplies
 * the DRAM behind `io.axi` and nothing else.
 */
class SimSoc(target: SystemTarget, diffTest: Boolean, mainOnly: Boolean = false) extends Module {

  val loaded = ChipLoader.load(target.pb)

  val topology = loaded.copy(tiles = loaded.tiles.filter(tile => !mainOnly || tile.main).map(tile =>
    tile.copy(cores = tile.cores.map {
      case RocketTileCore(cpu, Some(config)) =>
        RocketTileCore(cpu, Some(config.copy(sim = config.sim.copy(diffTest = diffTest))))
      case AntTileCore(local, config)        =>
        AntTileCore(local, config.copy(sim = config.sim.copy(diffTest = diffTest)))
      case core                              => core
    })
  ))

  val scu = SCUParams()

  val chi = ChiParams()

  // Per-tile template (System sets each tile's agent count). Tags and directories are still
  // registers in these IPs; sizes stay modest until they move to SRAM.
  val memory = CoherenceParams(
    chi,
    CacheParams(chi.addressBits, 64, 64, 4, 8, 4, 2),
    agents = 2,
    mshrEntries = 8,
    homeId = math.max(64, 2 * topology.tiles.map(_.hartIds.size).sum + 1)
  )

  // Every hart starts in the BootROM, which validates mhartid and jumps to the DRAM entry.
  val hartIdList = topology.tiles.flatMap(_.hartIds)
  require(hartIdList.sorted == hartIdList.indices, "BootROM validates harts as 0 until the hart count")
  val bootrom    = BootRom(BigInt("10000", 16), BigInt("10000", 16), hartIdList.size, target.dramBase)
  val devices    = DeviceParams(bootrom = Some(bootrom), scu = scu)

  val regions = Seq(
    PhysicalRegion(target.dramBase, target.dramBytes, true, true, true, true, true, true),
    PhysicalRegion(bootrom.base, bootrom.bytes, true, true, true, false, false, true),
    PhysicalRegion(devices.clint.base, devices.clint.bytes, false, false, true, true, false, false),
    PhysicalRegion(devices.plic.base, devices.plic.bytes, false, false, true, true, false, false),
    PhysicalRegion(scu.baseAddress, scu.totalSizeBytes, false, false, true, true, false, false)
  )

  val slotsPerClient = 16

  val ddr = DdrParams(
    chi,
    clients = 1,
    slotsPerClient = slotsPerClient,
    dataBits = 128,
    idBits = 4
  )

  val axiParams: axi4.Params = ddr.axi

  val io = IO(new Bundle {
    val axi = new axi4.Port(axiParams)
  })

  val system: Instance[System] = target.instantiate(SystemParams(
    topology,
    36,
    memory,
    RnfParams(chi, cacheLines = 64, banks = 2, homeId = memory.homeId),
    regions,
    TrackingParams(addressBits = chi.addressBits),
    ddr,
    devices
  ))

  io.axi <> system.dlink.axi
}
