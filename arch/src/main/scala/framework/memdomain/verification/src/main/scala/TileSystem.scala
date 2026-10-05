package framework.memdomain.verification

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.system.configloader.ExampleTopology
import framework.system.tile.Tile
import framework.system.memory.Memory
import framework.memdomain.frontend.mem.dma.DmaStatus
import hier.tile.memory.CoreInterrupts
import memcore.bus.chi.rnf.RnfParams
import memcore.memory.coherence.configs.CoherenceParams
import memcore.memory.cpu.{CpuMemParams, PhysicalRegion, UncachedRequest, UncachedResponse}
import memcore.memory.interlock.{Params => TrackingParams}
import memcore.memory.uncached_ram.{Params => RamParams}
import memcore.memory.ddr.{Params => DdrParams}

/** Verification names the same production Tile/DDR composition. */
@instantiable
class TileSystem(
  topology:        ExampleTopology,
  cpuPhysicalBits: Int,
  memory:          CoherenceParams,
  l1:              RnfParams,
  regions:         Seq[PhysicalRegion],
  tracking:        TrackingParams,
  ram:             RamParams,
  ddr:             DdrParams)
    extends Module {
  val system: Instance[framework.system.System] =
    Instantiate(new framework.system.System(topology, cpuPhysicalBits, memory, l1, regions, tracking, ram, ddr))
  @public
  val io = IO(chiselTypeOf(system.io))
  io <> system.io
}
