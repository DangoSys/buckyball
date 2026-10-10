package framework.system.core.rocket.configs {

  import upickle.default._
  import freechips.rocketchip.rocket.{BHTParams, BTBParams, DCacheParams, ICacheParams, MulDivParams, RocketCoreParams}
  import freechips.rocketchip.tile.FPUParams

  /**
   * JSON-serializable Rocket CPU configuration parameters.
   *
   * Optional features (mulDiv, fpu, btb) use an `enable` flag instead of Option[T],
   * so JSON stays as a plain dictionary (upickle serializes Option as array).
   */

  case class MulDivParam(
    divUnroll:              Int,
    divEarlyOutGranularity: Int,
    enable:                 Boolean,
    mulUnroll:              Int,
    mulEarlyOut:            Boolean,
    divEarlyOut:            Boolean)

  object MulDivParam {
    implicit val rw: ReadWriter[MulDivParam] = macroRW

    def toMulDivParams(p: MulDivParam): Option[MulDivParams] =
      if (p.enable) Some(MulDivParams(
        divUnroll = p.divUnroll,
        divEarlyOutGranularity = p.divEarlyOutGranularity,
        mulUnroll = p.mulUnroll,
        mulEarlyOut = p.mulEarlyOut,
        divEarlyOut = p.divEarlyOut
      ))
      else None

  }

  case class FPUParam(
    divSqrt:     Boolean,
    sfmaLatency: Int,
    dfmaLatency: Int,
    fpmuLatency: Int,
    ifpuLatency: Int,
    enable:      Boolean,
    minFLen:     Int,
    fLen:        Int)

  object FPUParam {
    implicit val rw: ReadWriter[FPUParam] = macroRW

    def toFPUParams(p: FPUParam): Option[FPUParams] =
      if (p.enable) Some(FPUParams(
        divSqrt = p.divSqrt,
        sfmaLatency = p.sfmaLatency,
        dfmaLatency = p.dfmaLatency,
        fpmuLatency = p.fpmuLatency,
        ifpuLatency = p.ifpuLatency,
        minFLen = p.minFLen,
        fLen = p.fLen
      ))
      else None

  }

  case class DCacheParam(
    clockGate: Boolean,
    nSets:     Int,
    nWays:     Int,
    nMSHRs:    Int)

  object DCacheParam {
    implicit val rw: ReadWriter[DCacheParam] = macroRW

    def toDCacheParams(p: DCacheParam, rowBits: Int, blockBytes: Int): DCacheParams = DCacheParams(
      clockGate = p.clockGate,
      nSets = p.nSets,
      nWays = p.nWays,
      rowBits = rowBits,
      nMSHRs = p.nMSHRs,
      blockBytes = blockBytes
    )

  }

  case class ICacheParam(
    prefetch: Boolean,
    nSets:    Int,
    nWays:    Int)

  object ICacheParam {
    implicit val rw: ReadWriter[ICacheParam] = macroRW

    def toICacheParams(p: ICacheParam, rowBits: Int, blockBytes: Int): ICacheParams = ICacheParams(
      prefetch = p.prefetch,
      nSets = p.nSets,
      nWays = p.nWays,
      rowBits = rowBits,
      blockBytes = blockBytes
    )

  }

  case class BTBParam(
    nMatchBits:        Int,
    nPages:            Int,
    updatesOutOfOrder: Boolean,
    bhtEnable:         Boolean,
    bhtEntries:        Int,
    bhtCounterLength:  Int,
    bhtHistoryLength:  Int,
    bhtHistoryBits:    Int,
    enable:            Boolean,
    nEntries:          Int,
    nRAS:              Int)

  object BTBParam {
    implicit val rw: ReadWriter[BTBParam] = macroRW

    def toBTBParams(p: BTBParam): Option[BTBParams] =
      if (p.enable) Some(BTBParams(
        nMatchBits = p.nMatchBits,
        nPages = p.nPages,
        updatesOutOfOrder = p.updatesOutOfOrder,
        bhtParams =
          if (p.bhtEnable) Some(BHTParams(p.bhtEntries, p.bhtCounterLength, p.bhtHistoryLength, p.bhtHistoryBits))
          else None,
        nEntries = p.nEntries,
        nRAS = p.nRAS
      ))
      else None

  }

  case class RocketCpuParam(
    mtvecInitEnable:         Boolean,
    useUser:                 Boolean,
    useSupervisor:           Boolean,
    useHypervisor:           Boolean,
    useDebug:                Boolean,
    useAtomics:              Boolean,
    useAtomicsOnlyForIO:     Boolean,
    useCompressed:           Boolean,
    useRVE:                  Boolean,
    useConditionalZero:      Boolean,
    nLocalInterrupts:        Int,
    useNMI:                  Boolean,
    nBreakpoints:            Int,
    useBPWatch:              Boolean,
    mcontextWidth:           Int,
    scontextWidth:           Int,
    nPerfCounters:           Int,
    haveBasicCounters:       Boolean,
    misaWritable:            Boolean,
    mtvecInit:               BigInt,
    mtvecWritable:           Boolean,
    fastLoadWord:            Boolean,
    fastLoadByte:            Boolean,
    branchPredictionModeCSR: Boolean,
    clockGate:               Boolean,
    mvendorid:               Int,
    mimpid:                  Int,
    haveCease:               Boolean,
    haveSimTimeout:          Boolean,
    useVM:                   Boolean,
    useZba:                  Boolean,
    useZbb:                  Boolean,
    useZbs:                  Boolean,
    haveCFlush:              Boolean,
    mulDiv:                  MulDivParam,
    fpu:                     FPUParam,
    dcache:                  DCacheParam,
    icache:                  ICacheParam,
    btb:                     BTBParam)

  object RocketCpuParam {
    implicit val rw: ReadWriter[RocketCpuParam] = macroRW

    /**
     * Convert to rocket-chip RocketCoreParams.
     * Cache geometry, physical address width and bus widths are supplied explicitly by the core.
     */
    def toRocketCoreParams(p: RocketCpuParam, xLen: Int, pgLevels: Int): RocketCoreParams = RocketCoreParams(
      xLen = xLen,
      pgLevels = pgLevels,
      useUser = p.useUser,
      useSupervisor = p.useSupervisor,
      useHypervisor = p.useHypervisor,
      useDebug = p.useDebug,
      useAtomics = p.useAtomics,
      useAtomicsOnlyForIO = p.useAtomicsOnlyForIO,
      useCompressed = p.useCompressed,
      useRVE = p.useRVE,
      useConditionalZero = p.useConditionalZero,
      nLocalInterrupts = p.nLocalInterrupts,
      useNMI = p.useNMI,
      nBreakpoints = p.nBreakpoints,
      useBPWatch = p.useBPWatch,
      mcontextWidth = p.mcontextWidth,
      scontextWidth = p.scontextWidth,
      nPerfCounters = p.nPerfCounters,
      haveBasicCounters = p.haveBasicCounters,
      misaWritable = p.misaWritable,
      mtvecInit = if (p.mtvecInitEnable) Some(p.mtvecInit) else None,
      mtvecWritable = p.mtvecWritable,
      fastLoadWord = p.fastLoadWord,
      fastLoadByte = p.fastLoadByte,
      branchPredictionModeCSR = p.branchPredictionModeCSR,
      clockGate = p.clockGate,
      mvendorid = p.mvendorid,
      mimpid = p.mimpid,
      haveCease = p.haveCease,
      haveSimTimeout = p.haveSimTimeout,
      useVM = p.useVM,
      useZba = p.useZba,
      useZbb = p.useZbb,
      useZbs = p.useZbs,
      haveCFlush = p.haveCFlush,
      mulDiv = MulDivParam.toMulDivParams(p.mulDiv),
      fpu = FPUParam.toFPUParams(p.fpu)
    )

    def toDCacheParams(p: RocketCpuParam, rowBits: Int, blockBytes: Int): DCacheParams =
      DCacheParam.toDCacheParams(p.dcache, rowBits, blockBytes)

    def toICacheParams(p: RocketCpuParam, rowBits: Int, blockBytes: Int): ICacheParams =
      ICacheParam.toICacheParams(p.icache, rowBits, blockBytes)

    def toBTBParams(p: RocketCpuParam): Option[BTBParams] =
      BTBParam.toBTBParams(p.btb)
  }

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
