package sims.soc

import chisel3._
import chisel3.util.PriorityEncoder
import chisel3.experimental.hierarchy.Instantiate
import chisel3.experimental.hierarchy.Instance
import framework.system.System
import framework.system.configloader.{ChipLoader, ExampleTopology, RocketTileCore}
import framework.system.device.{BootRom, DeviceParams}
import memcore.bus.axi4
import memcore.bus.chi.{Params => ChiParams}
import memcore.bus.chi.rnf.RnfParams
import memcore.memory.cache.configs.CacheParams
import memcore.memory.coherence.configs.CoherenceParams
import memcore.memory.cpu.{CpuMemParams, PhysicalRegion}
import memcore.memory.ddr.{Params => DdrParams}
import memcore.memory.interlock.{Params => TrackingParams}
import memcore.memory.uncached_ram.{Params => RamParams}
import sims.scu.{SCUParams, SystemControl}

/** A chip on the explicit System: one PB plus the platform's DRAM window. */
abstract class SystemTarget(val pb: String) {
  val dramBase:  BigInt = BigInt("80000000", 16)
  val dramBytes: BigInt = BigInt(512) << 20
}

/**
 * The explicit System with its SCU and NPU-fault abort. Every platform (Verilator, P2E) supplies
 * the DRAM behind `io.axi` and nothing else.
 */
class SimSoc(target: SystemTarget, diffTest: Boolean) extends Module {

  private val topology = ExampleTopology(ChipLoader.load(target.pb).tiles.map(tile =>
    tile.copy(cores = tile.cores.map {
      case RocketTileCore(cpu, Some(config)) =>
        RocketTileCore(cpu, Some(config.copy(sim = config.sim.copy(diffTest = diffTest))))
      case core                              => core
    })
  ))

  private val cores = topology.tiles.map(_.cores.size).sum

  private val scu = SCUParams()

  private val chi = ChiParams()

  // Per-tile template (System sets each tile's agent count). Tags and directories are still
  // registers in these IPs; sizes stay modest until they move to SRAM.
  private val memory = CoherenceParams(
    chi,
    CacheParams(chi.addressBits, 64, 64, 4, 8, 4, 2),
    agents = 2,
    mshrEntries = 8,
    homeId = math.max(64, 2 * topology.tiles.map(_.cores.size).max + 1)
  )

  // Every hart starts in the BootROM, which validates mhartid and jumps to the DRAM entry.
  private val hartIdList = topology.tiles.flatMap(_.hartIds)
  require(hartIdList.sorted == hartIdList.indices, "BootROM validates harts as 0 until the hart count")
  private val bootrom    = BootRom(BigInt("10000", 16), BigInt("10000", 16), hartIdList.size, target.dramBase)
  private val devices    = DeviceParams(bootrom = Some(bootrom))

  private val regions = Seq(
    PhysicalRegion(target.dramBase, target.dramBytes, true, true, true, true, true, true),
    PhysicalRegion(bootrom.base, bootrom.bytes, true, true, true, false, false, true),
    PhysicalRegion(devices.clint.base, devices.clint.bytes, false, false, true, true, false, false),
    PhysicalRegion(devices.plic.base, devices.plic.bytes, false, false, true, true, false, false),
    PhysicalRegion(scu.baseAddress, scu.totalSizeBytes, false, false, true, true, false, false)
  )

  private val ram = RamParams(
    chi,
    cpuHartIds = hartIdList.map(BigInt(_)),
    lineAgents = topology.tiles.size,
    slots = 16,
    base = target.dramBase,
    bytes = target.dramBytes
  )

  // The NPU DMA shares this port, so its data width follows the Buckyball DMA bus.
  private val ddr = DdrParams(chi, clients = 2, slotsPerClient = 8, dataBits = 128, idBits = 4)
  val axiParams: axi4.Params = ddr.axi

  val io = IO(new Bundle {
    val axi = new axi4.Port(axiParams)
  })

  val system: Instance[System] = Instantiate(new System(
    topology,
    36,
    memory,
    RnfParams(chi, cacheLines = 64, banks = 2, homeId = memory.homeId),
    regions,
    TrackingParams(addressBits = chi.addressBits),
    ram,
    ddr,
    devices
  ))

  // A halted Admission stalls its core forever: report the first fault and end the run with
  // exit code 0x200 | error, beside the 0x100 | mcause trap codes the task fixtures use.
  private val failed     = RegInit(false.B)
  private val anyFailure = system.io.failure.map(_.valid).reduce(_ || _)
  private val firstCore  = PriorityEncoder(system.io.failure.map(_.valid))
  private val hartIds    = VecInit(hartIdList.map(_.U(32.W)))
  for ((failure, core) <- system.io.failure.zipWithIndex) {
    when(failure.valid && !failed && firstCore === core.U) {
      printf(cf"[SimSoc] core $core NPU failure: error=${failure.bits.error} address=0x${failure.bits.address}%x\n")
    }
  }
  when(anyFailure)(failed := true.B)

  system.io.resetVector.foreach(_ := bootrom.base.U)
  // The console is polled through SBI; no device drives a PLIC source yet.
  system.io.interruptSources      := 0.U

  val control: Instance[SystemControl] = Instantiate(new SystemControl(cores, CpuMemParams(chi, ram.tagBits), scu))
  control.io.request <> system.io.deviceRequest
  control.io.abort.valid     := anyFailure && !failed
  control.io.abort.bits.hart := hartIds(firstCore)
  control.io.abort.bits.code := 0x200.U | VecInit(system.io.failure.map(_.bits.error))(firstCore)
  system.io.deviceResponse <> control.io.response

  io.axi <> system.io.axi
}
