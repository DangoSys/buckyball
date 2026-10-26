// See LICENSE.SiFive for license details.

package freechips.rocketchip.tile {

  import chisel3._
  import chisel3.util.isPow2
  import org.chipsalliance.cde.config._
  import freechips.rocketchip.rocket._
  import freechips.rocketchip.util._

  case object MaxHartIdBits extends Field[Int]
  case object CorePAddrBits extends Field[Int]

// These parameters can be varied per-core
  trait CoreParams {
    val bootFreqHz:          BigInt
    val useVM:               Boolean
    val useHypervisor:       Boolean
    val useUser:             Boolean
    val useSupervisor:       Boolean
    val useDebug:            Boolean
    val useAtomics:          Boolean
    val useAtomicsOnlyForIO: Boolean
    val useCompressed:       Boolean
    val useVector:       Boolean = false
    val vectorUseDCache: Boolean = false
    val useRVE:             Boolean
    val useConditionalZero: Boolean
    val useZba:             Boolean
    val useZbb:             Boolean
    val useZbs:             Boolean
    val mulDiv:             Option[MulDivParams]
    val fpu:                Option[FPUParams]
    val fetchWidth:         Int
    val decodeWidth:        Int
    val retireWidth:        Int
    val instBits:           Int
    val nLocalInterrupts:   Int
    val useNMI:             Boolean
    val nPMPs:              Int
    val pmpGranularity:     Int
    val nBreakpoints:       Int
    val useBPWatch:         Boolean
    val mcontextWidth:      Int
    val scontextWidth:      Int
    val nPerfCounters:      Int
    val haveBasicCounters:  Boolean
    val haveFSDirty:        Boolean
    val misaWritable:       Boolean
    val haveCFlush:         Boolean
    val nL2TLBEntries:      Int
    val nL2TLBWays:         Int
    val nPTECacheEntries:   Int
    val mtvecInit:          Option[BigInt]
    val mtvecWritable:      Boolean
    val traceHasWdata:      Boolean
    val xLen:               Int
    val pgLevels:           Int
    def traceCustom:  Option[Data]   = None
    def customIsaExt: Option[String] = None
    def customCSRs(implicit p: Parameters): CustomCSRs = new CustomCSRs

    def hasSupervisorMode: Boolean = useSupervisor || useVM
    def instBytes:         Int     = instBits / 8
    def fetchBytes:        Int     = fetchWidth * instBytes
    def lrscCycles: Int

    def dcacheReqTagBits: Int = 6

    def minFLen: Int = 32

    def vLen:         Int         = 0
    def eLen:         Int         = 0
    def vfLen:        Int         = 0
    def vfh:          Boolean     = false
    def vExts:        Seq[String] = Nil
    def hasV:         Boolean     = vLen >= 128 && eLen >= 64 && vfLen >= 64
    def vMemDataBits: Int         = 0

    def useBitmanip = useZba && useZbb && useZbs
  }

  trait HasCoreParameters extends HasTileParameters {

    override lazy val paddrBits: Int = {
      val bits = p(CorePAddrBits)
      require(bits <= maxPAddrBits)
      bits
    }

    val coreParams: CoreParams = tileParams.core

    val minFLen = coreParams.fpu.map(_ => coreParams.minFLen).getOrElse(0)
    val fLen    = coreParams.fpu.map(_.fLen).getOrElse(0)

    val usingMulDiv           = coreParams.mulDiv.nonEmpty
    val usingFPU              = coreParams.fpu.nonEmpty
    val usingAtomics          = coreParams.useAtomics
    val usingAtomicsOnlyForIO = coreParams.useAtomicsOnlyForIO
    val usingAtomicsInCache   = usingAtomics && !usingAtomicsOnlyForIO
    val usingCompressed       = coreParams.useCompressed
    val usingVector           = coreParams.useVector
    val usingNMI              = coreParams.useNMI
    val usingConditionalZero  = coreParams.useConditionalZero

    val retireWidth = coreParams.retireWidth
    val fetchWidth  = coreParams.fetchWidth
    val decodeWidth = coreParams.decodeWidth

    val fetchBytes      = coreParams.fetchBytes
    val coreInstBits    = coreParams.instBits
    val coreInstBytes   = coreInstBits / 8
    val coreDataBits    = xLen max fLen max vMemDataBits
    val coreDataBytes   = coreDataBits / 8
    def coreMaxAddrBits = paddrBits max vaddrBitsExtended

    val nBreakpoints   = coreParams.nBreakpoints
    val nPMPs          = coreParams.nPMPs
    val pmpGranularity = coreParams.pmpGranularity
    val nPerfCounters  = coreParams.nPerfCounters
    val mtvecInit      = coreParams.mtvecInit
    val mtvecWritable  = coreParams.mtvecWritable
    val customIsaExt   = coreParams.customIsaExt
    val traceHasWdata  = coreParams.traceHasWdata

    def vLen         = coreParams.vLen
    def eLen         = coreParams.eLen
    def vfLen        = coreParams.vfLen
    def vMemDataBits = if (usingVector) coreParams.vMemDataBits else 0
    def maxVLMax     = vLen

    if (usingVector) {
      require(isPow2(vLen), s"vLen ($vLen) must be a power of 2")
      require(eLen >= 32 && vLen % eLen == 0, s"eLen must divide vLen ($vLen) and be no less than 32")
      require(eLen == 32 || eLen == 64)
      require(vfLen <= eLen)
      require(!coreParams.vfh || (vfLen >= 32 && coreParams.minFLen <= 16))
    }

    if (coreParams.useVM) {
      if (coreParams.xLen == 32) {
        require(coreParams.pgLevels == 2)
      } else {
        require(coreParams.pgLevels >= 3)
      }
    }

    lazy val hartIdLen: Int = p(MaxHartIdBits)

    lazy val resetVectorLen: Int = {
      val externalLen = paddrBits
      require(externalLen <= xLen, s"External reset vector length ($externalLen) must be <= XLEN ($xLen)")
      require(
        externalLen <= vaddrBitsExtended,
        s"External reset vector length ($externalLen) must be <= virtual address bit width ($vaddrBitsExtended)"
      )
      externalLen
    }

    // Print out log of committed instructions and their writeback values.
    // Requires post-processing due to out-of-order writebacks.
    val enableCommitLog = false

  }

  abstract class CoreModule(implicit val p: Parameters) extends Module with HasCoreParameters

  abstract class CoreBundle(implicit val p: Parameters) extends ParameterizedBundle()(p) with HasCoreParameters

// This is a raw commit trace from the core, not the TraceCoreInterface
  class TraceBundle(implicit val p: Parameters) extends Bundle with HasCoreParameters {
    val insns  = Vec(coreParams.retireWidth, new TracedInstruction)
    val time   = UInt(64.W)
    val custom = coreParams.traceCustom
  }

}

package examples.core.rocket.pipeline {

  import freechips.rocketchip.rocket.{BTBParams, DCacheParams, ICacheParams, RocketCoreParams}

  /** Explicit CPU metadata; the upstream parameter conversion stays in the Rocket adapter. */
  case class CpuParams(
    core:                RocketCoreParams,
    dcache:              Option[DCacheParams],
    icache:              Option[ICacheParams],
    btb:                 Option[BTBParams],
    physicalAddressBits: Int,
    hartIdBits:          Int,
    beatBytes:           Int = 8,
    blockBytes:          Int = 64,
    asIdBits:            Int = 0)

}

package examples.core.rocket.pipeline {

  import org.chipsalliance.cde.config.Parameters
  import freechips.rocketchip.tile.{
    CoreBeatBytes,
    CorePAddrBits,
    CpuTileParams,
    HasCoreParameters,
    MaxHartIdBits,
    TileKey
  }
  import freechips.rocketchip.rocket.ASIdBits
  import freechips.rocketchip.subsystem.CacheBlockBytes

  object RocketParameters {

    def apply(cpu: CpuParams): Parameters = Parameters.empty.alterPartial {
      case TileKey         => CpuTileParams(core = cpu.core, dcache = cpu.dcache, icache = cpu.icache, btb = cpu.btb)
      case CorePAddrBits   => cpu.physicalAddressBits
      case MaxHartIdBits   => cpu.hartIdBits
      case CoreBeatBytes   => cpu.beatBytes
      case CacheBlockBytes => cpu.blockBytes
      case ASIdBits        => cpu.asIdBits
    }

  }

  trait HasCpuParameters extends HasCoreParameters {
    implicit val cpuParams: CpuParams
    implicit lazy val p: Parameters = RocketParameters(cpuParams)
  }

}

// See LICENSE.Berkeley and LICENSE.SiFive for Rocket CSR semantics.
package freechips.rocketchip.rocket {

  import chisel3._
  import org.chipsalliance.cde.config.Parameters
  import freechips.rocketchip.tile._
  import freechips.rocketchip.util._

  trait HasRocketCoreParameters extends HasCoreParameters {
    lazy val rocketParams = tileParams.core.asInstanceOf[RocketCoreParams]
    val fastLoadWord      = rocketParams.fastLoadWord
    val fastLoadByte      = rocketParams.fastLoadByte
    val mulDivParams      = rocketParams.mulDiv.getOrElse(MulDivParams())
    require(!fastLoadByte || fastLoadWord)
    require(!rocketParams.haveFSDirty)
  }

  class RocketCustomCSRs(implicit p: Parameters) extends CustomCSRs with HasRocketCoreParameters {
    override def bpmCSR = rocketParams.branchPredictionModeCSR.option(CustomCSR(bpmCSRId, 1, Some(BigInt(0))))

    override def chickenCSR = {
      val dc   = tileParams.dcache.get
      val mask = BigInt(dc.clockGate.toInt | (rocketParams.clockGate.toInt << 1) |
        (rocketParams.clockGate.toInt << 2) | (1 << 3) | (dc.scratch.isEmpty.toInt << 9) |
        (tileParams.icache.get.prefetch.toInt << 17))
      Some(CustomCSR(chickenCSRId, mask, Some(mask)))
    }

    def disableICachePrefetch = getOrElse(chickenCSR, _.value(17), true.B)
    def marchid               = CustomCSR.constant(CSRs.marchid, BigInt(1))
    def mvendorid             = CustomCSR.constant(CSRs.mvendorid, BigInt(rocketParams.mvendorid))
    def mimpid                = CustomCSR.constant(CSRs.mimpid, BigInt(rocketParams.mimpid))
    override def decls        = super.decls :+ marchid :+ mvendorid :+ mimpid
  }

}

// See LICENSE.Berkeley and LICENSE.SiFive for Rocket interface definitions.
package freechips.rocketchip.tile {

  import org.chipsalliance.cde.config.{Field, Parameters}
  import chisel3.util.{log2Ceil, log2Up}
  import freechips.rocketchip.rocket._

  case object TileKey       extends Field[CpuTileParams]
  case object CoreBeatBytes extends Field[Int]
  case object CoreRoCCCount extends Field[Int](0)

  trait HasTileParameters {
    implicit val p: Parameters
    def tileParams: CpuTileParams = p(TileKey)
    def usingVM                    = tileParams.core.useVM
    def usingSupervisor            = tileParams.core.hasSupervisorMode
    def usingUser                  = tileParams.core.useUser || usingSupervisor
    def usingHypervisor            = usingVM && tileParams.core.useHypervisor
    def usingDebug                 = tileParams.core.useDebug
    def usingRoCC                  = p(CoreRoCCCount) > 0
    def usingBTB                   = tileParams.btb.exists(_.nEntries > 0)
    def usingPTW                   = usingVM
    def usingDataScratchpad        = tileParams.dcache.flatMap(_.scratch).isDefined
    def xLen                       = tileParams.core.xLen
    def xBytes                     = xLen / 8
    def iLen                       = 32
    def pgIdxBits                  = 12
    def pgLevelBits                = 10 - log2Ceil(xLen / 32)
    def pgLevels                   = tileParams.core.pgLevels
    def minPgLevels = { val levels = if (xLen == 32) 2 else 3; require(pgLevels >= levels); levels }
    def maxSVAddrBits              = pgIdxBits + pgLevels * pgLevelBits
    def maxHypervisorExtraAddrBits = 2
    def hypervisorExtraAddrBits    = if (usingHypervisor) maxHypervisorExtraAddrBits else 0
    def maxHVAddrBits              = maxSVAddrBits + hypervisorExtraAddrBits
    def asIdBits                   = p(ASIdBits)
    def vmIdBits                   = p(VMIdBits)

    lazy val maxPAddrBits: Int = {
      require(xLen == 32 || xLen == 64); if (!usingVM) xLen else if (xLen == 32) 34 else 56
    }

    lazy val paddrBits: Int = p(CorePAddrBits)

    def vaddrBits: Int =
      if (usingVM) {
        val bits = maxHVAddrBits
        require(bits == xLen || xLen > bits && bits > paddrBits)
        bits
      } else (paddrBits + 1).min(xLen)

    def vpnBits             = vaddrBits - pgIdxBits
    def ppnBits             = paddrBits - pgIdxBits
    def vpnBitsExtended     = vpnBits + (if (vaddrBits < xLen) 1 + (if (usingHypervisor) 1 else 0) else 0)
    def vaddrBitsExtended   = vpnBitsExtended + pgIdxBits
    def cacheBlockBytes     = p(freechips.rocketchip.subsystem.CacheBlockBytes)
    def lgCacheBlockBytes   = log2Up(cacheBlockBytes)
    def masterPortBeatBytes = p(CoreBeatBytes)
    def dcacheArbPorts      = 1 + (if (usingVM) 1 else 0) + (if (usingDataScratchpad) 1 else 0) + p(CoreRoCCCount) +
      (if (tileParams.core.useVector && tileParams.core.vectorUseDCache) 1 else 0)
  }

}

package freechips.rocketchip.rocket {
  import org.chipsalliance.cde.config.Field
  case object ASIdBits extends Field[Int](0)
  case object VMIdBits extends Field[Int](0)
}

package freechips.rocketchip.subsystem {
  import org.chipsalliance.cde.config.Field
  case object CacheBlockBytes extends Field[Int]
}

package freechips.rocketchip.devices.debug {
  import org.chipsalliance.cde.config.Field
  case object DebugModuleKey extends Field[Option[DebugModuleParams]](Some(DebugModuleParams()))
}

package freechips.rocketchip.rocket {
  import chisel3._
  import chisel3.util._
  import org.chipsalliance.cde.config.Parameters
  import freechips.rocketchip.tile._

  case class DCacheParams(
    nSets:                Int = 64,
    nWays:                Int = 4,
    rowBits:              Int = 64,
    subWordBits:          Option[Int] = None,
    replacementPolicy:    String = "random",
    nTLBSets:             Int = 1,
    nTLBWays:             Int = 32,
    nTLBBasePageSectors:  Int = 4,
    nTLBSuperpages:       Int = 4,
    tagECC:               Option[String] = None,
    dataECC:              Option[String] = None,
    dataECCBytes:         Int = 1,
    nMSHRs:               Int = 1,
    nSDQ:                 Int = 17,
    nRPQ:                 Int = 16,
    nMMIOs:               Int = 1,
    blockBytes:           Int = 64,
    separateUncachedResp: Boolean = false,
    acquireBeforeRelease: Boolean = false,
    pipelineWayMux:       Boolean = false,
    clockGate:            Boolean = false,
    scratch:              Option[BigInt] = None) {
    require(scratch.isEmpty || nWays == 1)
    require(scratch.isEmpty || nMSHRs == 0)
    if (scratch.isEmpty) require(isPow2(nSets))
  }

  case class ICacheParams(
    nSets:               Int = 64,
    nWays:               Int = 4,
    rowBits:             Int = 128,
    nTLBSets:            Int = 1,
    nTLBWays:            Int = 32,
    nTLBBasePageSectors: Int = 4,
    nTLBSuperpages:      Int = 4,
    cacheIdBits:         Int = 0,
    tagECC:              Option[String] = None,
    dataECC:             Option[String] = None,
    itimAddr:            Option[BigInt] = None,
    prefetch:            Boolean = false,
    blockBytes:          Int = 64,
    latency:             Int = 2,
    fetchBytes:          Int = 4)

  case class RocketCoreParams(
    xLen:                    Int = 64,
    pgLevels:                Int = 3,
    bootFreqHz:              BigInt = 0,
    useVM:                   Boolean = true,
    useUser:                 Boolean = false,
    useSupervisor:           Boolean = false,
    useHypervisor:           Boolean = false,
    useDebug:                Boolean = true,
    useAtomics:              Boolean = true,
    useAtomicsOnlyForIO:     Boolean = false,
    useCompressed:           Boolean = true,
    useRVE:                  Boolean = false,
    useConditionalZero:      Boolean = false,
    useZba:                  Boolean = false,
    useZbb:                  Boolean = false,
    useZbs:                  Boolean = false,
    nLocalInterrupts:        Int = 0,
    useNMI:                  Boolean = false,
    nBreakpoints:            Int = 1,
    useBPWatch:              Boolean = false,
    mcontextWidth:           Int = 0,
    scontextWidth:           Int = 0,
    nPMPs:                   Int = 8,
    nPerfCounters:           Int = 0,
    haveBasicCounters:       Boolean = true,
    haveCFlush:              Boolean = false,
    misaWritable:            Boolean = true,
    nL2TLBEntries:           Int = 0,
    nL2TLBWays:              Int = 1,
    nPTECacheEntries:        Int = 8,
    mtvecInit:               Option[BigInt] = Some(BigInt(0)),
    mtvecWritable:           Boolean = true,
    fastLoadWord:            Boolean = true,
    fastLoadByte:            Boolean = false,
    branchPredictionModeCSR: Boolean = false,
    clockGate:               Boolean = false,
    mvendorid:               Int = 0,
    mimpid:                  Int = 0x20181004,
    mulDiv:                  Option[MulDivParams] = Some(MulDivParams()),
    fpu:                     Option[FPUParams] = Some(FPUParams()),
    debugROB:                Option[DebugROBParams] = None,
    haveCease:               Boolean = true,
    haveSimTimeout:          Boolean = true,
    vector:                  Option[RocketCoreVectorParams] = None,
    enableTraceCoreIngress:  Boolean = false)
      extends CoreParams {
    val lgPauseCycles    = 5
    val haveFSDirty      = false
    val pmpGranularity   = if (useHypervisor) 4096 else 4
    val fetchWidth       = if (useCompressed) 2 else 1
    val decodeWidth      = fetchWidth / (if (useCompressed) 2 else 1)
    val retireWidth      = 1
    val instBits         = if (useCompressed) 16 else 32
    val lrscCycles       = 80
    val traceHasWdata    = debugROB.isDefined
    override def minFLen = fpu.map(_.minFLen).getOrElse(32)
    override def customCSRs(implicit p: Parameters) = new RocketCustomCSRs
    override val useVector       = vector.isDefined
    override val vectorUseDCache = vector.exists(_.useDCache)
    override def vLen            = vector.map(_.vLen).getOrElse(0)
    override def eLen            = vector.map(_.eLen).getOrElse(0)
    override def vfLen           = vector.map(_.vfLen).getOrElse(0)
    override def vfh             = vector.exists(_.vfh)
    override def vMemDataBits    = vector.map(_.vMemDataBits).getOrElse(0)
    override def vExts           = vector.map(_.vExts).getOrElse(Nil)
  }

  case class RocketCoreVectorParams(
    vLen:         Int,
    eLen:         Int,
    vfLen:        Int,
    vfh:          Boolean,
    vMemDataBits: Int,
    decoder:      Parameters => RocketVectorDecoder,
    useDCache:    Boolean,
    issueVConfig: Boolean,
    vExts:        Seq[String])

}

package freechips.rocketchip.tile {
  import freechips.rocketchip.rocket._

  case class CpuTileParams(
    core:   CoreParams,
    dcache: Option[DCacheParams],
    icache: Option[ICacheParams],
    btb:    Option[BTBParams])

}

package freechips.rocketchip.devices.debug {
  case class DebugModuleParams(nDscratch: Int = 1, debugEntry: BigInt = 0x800, debugException: BigInt = 0x808)
}
