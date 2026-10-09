package framework.memdomain.verification

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.system.configloader.ExampleTopology
import framework.system.{System, SystemParams}
import framework.system.memory.Memory
import framework.memdomain.frontend.mem.dma.DmaStatus
import hier.tile.memory.CoreInterrupts
import memcore.bus.chi.rnf.RnfParams
import memcore.memory.coherence.configs.CoherenceParams
import memcore.memory.cpu.{CpuMemParams, PhysicalRegion, UncachedRequest, UncachedResponse}
import memcore.memory.interlock.{Params => TrackingParams}
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
  ddr:             DdrParams,
  build:           SystemParams => Instance[System])
    extends Module {
  val system: Instance[framework.system.System] =
    build(SystemParams(topology, cpuPhysicalBits, memory, l1, regions, tracking, ddr))
  @public
  val io = IO(chiselTypeOf(system.dlink))
  io <> system.dlink
}
