package examples.hart.rocket.configs

import framework.generic.contract.memory.access.AccessParams

import framework.generic.contract.memory.access.PhysicalRegion

import framework.generic.acc.configs.AccParam
import framework.generic.acc.memdomain.tomem.configs.{Params => TransportParam}
import examples.hart.rocket.pipeline.CpuParams
import framework.hart.hlink.configs.{Params => HLinkParam}
import framework.mlp.fetch.configs.{Params => FetchParam}
import framework.hart.memory.configs.MemoryParam

import buckyball.config.HartPlacement
import buckyball.components.rocket.{Params => RocketProto}
import scala.jdk.CollectionConverters._
import framework.generic.bus.{axi4, chi}
import framework.generic.bus.chi.rnf.RnfParams
import framework.generic.memory.interlock.{Params => TrackingParam}
import framework.generic.memory.coherence.configs.CoherenceParams
import framework.hart.memory.home.configs.{Params => HomeParam}
import framework.hart.memory.access.cache.configs.{Params => CacheParam}
import framework.hart.memory.access.configs.{Params => AccessParam}
import freechips.rocketchip.rocket.{DCacheParams, ICacheParams, MulDivParams, RocketCoreParams}
import freechips.rocketchip.tile.FPUParams
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.acc.frontend.configs.FrontendParam
import framework.generic.acc.memdomain.configs.MemDomainParam
import framework.generic.acc.gpdomain.configs.GpDomainParam
import framework.generic.contract.configs.BallDomainParam

case class Params(
  cpu:          CpuParams,
  memory:       MemoryParam,
  fetch:        FetchParam,
  hlink:        HLinkParam,
  permission:   PmpParams,
  hartId:       Int,
  resetVector:  BigInt,
  mtimeAddress: BigInt,
  acc:          Option[AccParam]) {
  require(hlink.harts == 1 && !hlink.service)
  require(hlink.requesters == (if (hlink.backing) 0 else 2))
  require(cpu.core.xLen == 64 && !cpu.core.useHypervisor && cpu.core.useCompressed)
  require(cpu.core.nPMPs == permission.entries && cpu.physicalAddressBits == permission.addressBits)
  require(memory.memoryAccess.lineBits == 512 && memory.dcache.cache.probe)
  require(memory.threadPorts == 0)
}

object Params {

  def fromProto(hart: HartPlacement): Params = {
    require(hart.hasMemory && hart.hasImplementation && hart.hasSharedMem)
    val r   = hart.getImplementation.unpack(classOf[RocketProto])
    val bus = chi.Params.fromProto(hart.getMemory.getHome.getRequesters)
    val axi = axi4.Params.fromProto(hart.getMemory.getUncached.getAxi)
    val acc = Option.when(hart.hasAccelerator) {
      val a = hart.getAccelerator
      AccParam(
        FrontendParam.fromProto(a.getFrontend, hart.getParam.getXLen),
        MemDomainParam.fromProto(a.getMem, hart.getSharedMem, hart.getVirtualBankCount, 1),
        GpDomainParam.fromProto(a.getGpdomain),
        BallDomainParam.fromProto(a.getBalldomain),
        TransportParam.fromProto(a.getTransport)
      )
    }
    Params(
      CpuParams(
        RocketCoreParams(
          xLen = 64,
          pgLevels = r.getPageLevels,
          nPMPs = r.getPmpEntries,
          useVM = r.getVirtualMemory,
          useSupervisor = r.getSupervisor,
          useUser = r.getUser,
          useHypervisor = false,
          useDebug = false,
          useAtomics = r.getAtomics,
          useCompressed = true,
          // Local interrupt 16: a DMA page fault of the local accelerator.
          nLocalInterrupts = 1,
          clockGate = false,
          mulDiv =
            Option.when(r.getMultiplyDivide)(MulDivParams(mulUnroll = 8, mulEarlyOut = true, divEarlyOut = true)),
          fpu = Option.when(r.getFloatingPoint)(FPUParams())
        ),
        Some(DCacheParams()),
        Some(ICacheParams()),
        None,
        r.getPhysicalAddressBits,
        r.getHartIdBits
      ),
      MemoryParam.fromProto(hart.getMemory, 0),
      FetchParam(hart.getFetch.getInstructionBytes, hart.getMemory.getIcache.getLines),
      HLinkParam(
        1,
        !r.getExternalHome,
        if (r.getExternalHome) Seq(hart.getMemory.getIcache.getNodeId, hart.getMemory.getDcache.getCache.getNodeId)
        else Seq.empty,
        bus,
        axi,
        acc.map(_.transport.axi).getOrElse(axi4.Params.fromProto(hart.getInterconnect.getAxi)),
        false,
        false
      ),
      PmpParams(r.getPhysicalAddressBits, r.getPmpEntries),
      hart.getHartId,
      BigInt(java.lang.Long.toUnsignedString(hart.getResetVector)),
      BigInt(java.lang.Long.toUnsignedString(hart.getMtimeAddress)),
      acc
    )
  }

}
