package examples.core.rocket.configs

import examples.core.rocket.pipeline.CpuParams
import framework.core.clink.configs.{Params => CLinkParam}
import framework.core.execution.fetch.configs.{Params => FetchParam}
import framework.core.memory.configs.MemoryParam
import framework.core.memory.access.preflight.configs.PmpParams
import framework.acc.configs.AccParam

import buckyball.config.CorePlacement
import buckyball.components.rocket.{Params => RocketProto}
import scala.jdk.CollectionConverters._
import framework.ip.bus.{axi4, chi}
import framework.ip.bus.chi.rnf.RnfParams
import framework.ip.memory.interlock.{Params => TrackingParam}
import framework.ip.memory.coherence.configs.CoherenceParams
import framework.core.memory.home.configs.{Params => HomeParam}
import framework.core.memory.access.cache.configs.{Params => CacheParam}
import framework.core.memory.access.configs.{Params => AccessParam}
import framework.core.memory.access.{AccessParams, PhysicalRegion}
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.acc.gpdomain.configs.GpDomainParam
import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.memdomain.tomem.configs.{Params => TransportParam}
import freechips.rocketchip.rocket.{DCacheParams, ICacheParams, MulDivParams, RocketCoreParams}
import freechips.rocketchip.tile.FPUParams

case class Params(
  cpu:          CpuParams,
  memory:       MemoryParam,
  fetch:        FetchParam,
  clink:        CLinkParam,
  permission:   PmpParams,
  hartId:       Int,
  resetVector:  BigInt,
  mtimeAddress: BigInt,
  acc:          Option[AccParam]) {
  require(clink.harts == 1 && !clink.shared)
  require(clink.requesters == (if (clink.backing) 0 else 2))
  require(cpu.core.xLen == 64 && !cpu.core.useHypervisor && cpu.core.useCompressed)
  require(cpu.core.nPMPs == permission.entries && cpu.physicalAddressBits == permission.addressBits)
  require(memory.memoryAccess.lineBits == 0)
  require(memory.threadPorts == 0)
}

object Params {

  def fromProto(core: CorePlacement): Params = {
    require(core.hasMemory && core.hasImplementation && core.hasSharedMem)
    val r   = core.getImplementation.unpack(classOf[RocketProto])
    val bus = chi.Params.fromProto(core.getMemory.getHome.getRequesters)
    val axi = axi4.Params.fromProto(core.getMemory.getUncached.getAxi)
    val acc = Option.when(core.hasAccelerator) {
      val a = core.getAccelerator
      AccParam(
        FrontendParam.fromProto(a.getFrontend, core.getParam.getXLen, core.getHartId),
        MemDomainParam.fromProto(a.getMem, core.getSharedMem, core.getVirtualBankCount, 1),
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
          clockGate = false,
          mulDiv = Option.when(r.getMultiplyDivide)(MulDivParams()),
          fpu = Option.when(r.getFloatingPoint)(FPUParams())
        ),
        Some(DCacheParams()),
        Some(ICacheParams()),
        None,
        r.getPhysicalAddressBits,
        r.getHartIdBits
      ),
      MemoryParam.fromProto(core.getMemory, 0),
      FetchParam(core.getFetch.getInstructionBytes),
      CLinkParam(
        1,
        !r.getExternalHome,
        if (r.getExternalHome) Seq(core.getMemory.getIcache.getNodeId, core.getMemory.getDcache.getCache.getNodeId)
        else Seq.empty,
        bus,
        axi,
        acc.map(_.transport.axi).getOrElse(axi4.Params.fromProto(core.getInterconnect.getAxi)),
        None
      ),
      PmpParams(r.getPhysicalAddressBits, r.getPmpEntries),
      core.getHartId,
      BigInt(java.lang.Long.toUnsignedString(core.getResetVector)),
      BigInt(java.lang.Long.toUnsignedString(core.getMtimeAddress)),
      acc
    )
  }

}
