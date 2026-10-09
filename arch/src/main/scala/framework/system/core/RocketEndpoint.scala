package framework.system.core

import framework.system.GlobalConfigOps._
import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.system.core.clink.{CLink, HasCLink, NpuEvents, ShmPort}
import framework.system.core.rocket.CpuParams
import framework.top.GlobalConfig
import framework.memdomain.frontend.mem.dma.DmaPort
import framework.memdomain.backend.banks.btrace.PhysicalBankHash
import framework.memdomain.backend.shared.SharedMemLayout
import hier.core.rocket.{Commands, CpuCLink}
import memcore.bus.chi.rnf.RnfParams
import memcore.memory.cpu.PhysicalRegion

case class RocketCoreParams(
  cpu:         CpuParams,
  data:        RnfParams,
  instruction: RnfParams,
  regions:     Seq[PhysicalRegion],
  commands:    Commands,
  buckyball:   Option[GlobalConfig])

class AcceleratorCLink(b: GlobalConfig) extends Bundle {
  val hartId   = Input(UInt(64.W))
  val shmOwner = Input(UInt(64.W))
  val npu      = new NpuEvents(b)
  val shm      = new ShmPort(b)
  val mem      = new DmaPort(b.memDomain.dma_buswidth)

  val sharedHashes =
    if (b.sim.diffTest && b.memDomain.sharedEnable)
      Some(Input(Vec(SharedMemLayout.totalBank(b), new PhysicalBankHash(b))))
    else None

}

class RocketCLink(p: RocketCoreParams) extends CLink {
  val cpu         = new CpuCLink(p.data, Some(p.commands))(p.cpu)
  val accelerator = p.buckyball.map(new AcceleratorCLink(_))
}

/** Reusable implementation; the concrete design and class selection live in examples/cores. */
@instantiable
abstract class RocketEndpoint(b: framework.top.GlobalConfig) extends Module with HasCLink {
  val p             = b.rocketParams
  @public val clink = IO(new RocketCLink(p))
}
