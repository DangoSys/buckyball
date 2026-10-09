package framework.system.configloader

import buckyball.config.{
  BoomCpuConfig,
  Chip,
  CoreInstance,
  CpuConfig,
  FrontendConfig,
  MemDomainConfig,
  RocketCpuConfig,
  RvvConfig,
  SharedMemConfig,
  SpmConfig,
  TileKind,
  TileParamConfig,
  TilePlacement
}
import java.nio.file.{Files, Path, Paths}
import framework.balldomain.configs.{BallDomainParam, BallISAEntry, BallIdMapping}
import framework.frontend.configs.FrontendParam
import framework.rvv.configs.RvvParam
import framework.memdomain.configs.MemDomainParam
import framework.system.core.boom.configs.{BoomCpuParam, BoomDCacheParam, BoomICacheParam}
import framework.system.core.rocket.configs._
import framework.system.tile.configs.TileParam
import framework.top.GlobalConfig
import scala.jdk.CollectionConverters._

/** Load ExampleTopology from chip.pb. */
object ChipLoader {

  def load(pbPath: String): ExampleTopology = {
    val path    = Paths.get(pbPath)
    if (!Files.isRegularFile(path)) {
      throw new RuntimeException(s"chip.pb does not exist: $pbPath")
    }
    val repo    = repoRoot(path)
    val chip    = Chip.parseFrom(Files.readAllBytes(path))
    val cores   = chip.getCoresList.asScala.toSeq
    if (cores.isEmpty) {
      throw new RuntimeException(s"chip.pb has no cores: $pbPath")
    }
    val tiles   = chip.getTilesList.asScala.map(parseTile(_, cores, repo)).toSeq
    require(
      tiles.size == chip.getNTiles,
      s"chip.pb declares n_tiles=${chip.getNTiles} but defines ${tiles.size} tile(s) in $pbPath"
    )
    val hartIds = tiles.flatMap(_.hartIds)
    require(hartIds.forall(_ >= 0), "hart_id must fit a nonnegative Scala Int")
    require(hartIds.distinct.size == hartIds.size, "chip hart_id values must be globally unique")
    ExampleTopology(tiles)
  }

  def repoRoot(pb: Path): Path = {
    val abs    = pb.toAbsolutePath.normalize
    val s      = abs.toString
    val marker = "/examples/chips/"
    val i      = s.lastIndexOf(marker)
    if (i < 0) {
      throw new RuntimeException(s"chip.pb is not under examples/chips: $pb")
    }
    Paths.get(s.substring(0, i))
  }

  def repoFile(repo: Path, rel: String, what: String): String = {
    if (rel.isEmpty) {
      throw new RuntimeException(s"$what is empty")
    }
    val p        = Paths.get(rel)
    if (p.isAbsolute) {
      throw new RuntimeException(s"$what must be repo-relative: $rel")
    }
    val resolved = repo.resolve(p).normalize()
    if (!Files.isRegularFile(resolved)) {
      throw new RuntimeException(s"$what missing: $resolved")
    }
    resolved.toString
  }

  def parseTile(tile: TilePlacement, cores: Seq[CoreInstance], repo: Path): TileTopology = {
    val indices        = tile.getCoreIndicesList.asScala.map(_.toInt).toSeq
    val nCores         = indices.size
    val computeCoreIds =
      indices.zipWithIndex.collect { case (index, slot) if cores(index).getBalldomain.getBallNum > 0 => slot }
    val hasBuckyball   = computeCoreIds.nonEmpty
    val tileParam      = parseTileParam(tile.getParam)

    val shared           = tile.getSharedMem
    val virtualBankCount = tile.getVirtualBankCount
    if (hasBuckyball) {
      require(virtualBankCount > 0, s"tile ${tile.getPath}: virtual_bank_count must be > 0")
    }
    val sharedBankNum    =
      if (shared.getEnable) {
        require(shared.getBankWidth == 128, s"tile ${tile.getPath}: shared bank width must be 128 bits")
        indices.iterator.map(idx => cores(idx)).filter(_.getBalldomain.getBallNum > 0).foreach { core =>
          require(
            core.getMem.getBank.getWidth == shared.getBankWidth,
            s"tile ${tile.getPath}: all Buckyball cores must use shared bank width ${shared.getBankWidth}"
          )
        }
        require(shared.getEntries > 0, s"tile ${tile.getPath}: shared entries must be > 0")
        require(
          shared.getBankEntries > 0 && shared.getEntries % shared.getBankEntries == 0,
          s"tile ${tile.getPath}: shared entries ${shared.getEntries} must be divisible by shared bank entries ${shared.getBankEntries}"
        )
        shared.getEntries / shared.getBankEntries
      } else 0
    val tss              = Option.when(tile.hasTss)(parseSpm(tile.getTss))
    val tileCores        = indices.map { idx =>
      parseCore(cores(idx), tileParam, shared, sharedBankNum, virtualBankCount, nCores, computeCoreIds, tss, repo)
    }
    val ants             = indices.filter(i => cores(i).getCpu.getKind == "ant")
    ants.zipWithIndex.foreach { case (idx, context) =>
      require(
        !cores(idx).hasHartId && cores(idx).hasAntContextId && cores(idx).getAntContextId == context,
        s"core $idx: Ant requires explicit local context, without a CPU hart"
      )
    }
    val hartIds          = indices.filterNot(ants.contains).map { idx =>
      require(cores(idx).hasHartId && !cores(idx).hasAntContextId, s"core $idx: CPU requires hart_id only")
      cores(idx).getHartId
    }
    val main             = tile.getKind match {
      case TileKind.TILE_KIND_MAIN => true
      case TileKind.TILE_KIND_TILE => false
      case kind                    => throw new IllegalArgumentException(s"Unknown tile kind: $kind")
    }
    require(!main || !tile.hasControllerCoreIndex, "The main tile has no task controller")
    val controller       = Option.when(tile.hasControllerCoreIndex) {
      val local = indices.indexOf(tile.getControllerCoreIndex)
      require(local >= 0, s"tile ${tile.getPath}: controller must belong to the tile")
      local
    }
    require(tile.getFactoryClass.isEmpty == main, "Only mounted tiles must specify a factory")
    TileTopology(
      tileParam,
      main,
      tileCores,
      hartIds,
      indices,
      if (controller.isDefined) indices.map(idx => coreSignature(cores(idx), tss, shared)) else Nil,
      controller,
      Option.when(shared.getEnable)(SharedStorageParams(shared.getBankWidth, shared.getBankEntries, sharedBankNum)),
      indices.map(idx => cores(idx).getFactoryClass),
      tile.getFactoryClass
    )
  }

  def parseCore(
    core:             CoreInstance,
    tile:             TileParam,
    shared:           SharedMemConfig,
    sharedBankNum:    Int,
    virtualBankCount: Int,
    nCores:           Int,
    computeCoreIds:   Seq[Int],
    tss:              Option[memcore.memory.spm.Params],
    repo:             Path
  ): TileCore = {
    require(core.hasCpu, s"core ${core.getPkg}: missing cpu config")
    val cpu  = core.getCpu
    val kind = cpu.getKind
    kind match {
      case "rocket" =>
        require(cpu.hasRocket, s"core ${core.getPkg}: missing cpu.rocket")
        RocketTileCore(
          parseRocketCpu(cpu.getRocket),
          parseAccelerator(core, tile, shared, sharedBankNum, virtualBankCount, nCores, computeCoreIds, tss, repo)
        )
      case "ant"    =>
        require(cpu.hasAnt && tss.isDefined, s"core ${core.getPkg}: Ant requires local geometry and tile TSS")
        val a           = cpu.getAnt
        require(a.hasTls, s"core ${core.getPkg}: missing Ant TLS")
        val local       = framework.ant.Params(a.getCodeBytes, parseSpm(a.getTls), tss.get, a.getTaskBits)
        val accelerator =
          parseAccelerator(core, tile, shared, sharedBankNum, virtualBankCount, nCores, computeCoreIds, tss, repo)
        require(accelerator.isDefined, "A compute Ant must have an NPU endpoint")
        AntTileCore(local, accelerator.get)
      case "boom"   =>
        if (core.getBalldomain.getBallNum > 0) {
          throw new RuntimeException(s"core ${core.getPkg}: kind=boom forbids balldomain")
        }
        if (!cpu.hasBoom) {
          throw new RuntimeException(s"core ${core.getPkg}: kind=boom missing cpu.boom")
        }
        BoomTileCore(parseBoomCpu(cpu.getBoom))
      case other    =>
        throw new RuntimeException(s"core ${core.getPkg}: unsupported kind '$other'")
    }
  }

  def parseAccelerator(
    core:             CoreInstance,
    tile:             TileParam,
    shared:           SharedMemConfig,
    sharedBankNum:    Int,
    virtualBankCount: Int,
    nCores:           Int,
    computeCoreIds:   Seq[Int],
    tss:              Option[memcore.memory.spm.Params],
    repo:             Path
  ): Option[GlobalConfig] = {
    val domain   = core.getBalldomain
    if (domain.getBallNum == 0) {
      return None
    }
    require(core.hasFrontend, s"core ${core.getPkg} missing frontend config")
    require(core.hasRvv, s"core ${core.getPkg} missing rvv config")
    require(core.getRvv.hasEnable, s"core ${core.getPkg} must explicitly set rvv.enable")
    require(tile.coreDataBytes > 0, s"core ${core.getPkg} missing tile config")
    val frontend = core.getFrontend
    require(
      virtualBankCount <= (1 << frontend.getBankIdLen),
      s"core ${core.getPkg}: virtual_bank_count $virtualBankCount does not fit bank_id_len ${frontend.getBankIdLen}"
    )
    require(
      frontend.getVbankIdUpperBound < virtualBankCount,
      s"core ${core.getPkg}: private vbank upper bound ${frontend.getVbankIdUpperBound} must be below virtual_bank_count $virtualBankCount"
    )
    if (shared.getEnable) {
      require(
        frontend.getSharedBankIdBase > frontend.getVbankIdUpperBound,
        s"core ${core.getPkg}: shared bank base must be above the private vbank range"
      )
      require(
        frontend.getSharedBankIdBase <= virtualBankCount,
        s"core ${core.getPkg}: shared bank base ${frontend.getSharedBankIdBase} exceeds virtual_bank_count $virtualBankCount"
      )
    }

    val buckyball = GlobalConfig().copy(
      coreSignature = coreSignature(core, tss, shared),
      ballDomain = parseBallDomain(core, repo),
      frontend = parseFrontend(core.getFrontend),
      rvv = parseRvv(core.getRvv),
      tile = tile,
      memDomain = parseMemDomain(core.getMem, shared, sharedBankNum, virtualBankCount, nCores, computeCoreIds)
    )
    Some(buckyball)
  }

  def parseSpm(value: SpmConfig): memcore.memory.spm.Params =
    memcore.memory.spm.Params(BigInt(java.lang.Long.toUnsignedString(value.getBase)), value.getBytes, value.getDataBits)

  def coreSignature(
    core:         CoreInstance,
    tss:          Option[memcore.memory.spm.Params],
    sharedMemory: SharedMemConfig
  ): BigInt = {
    require(core.hasMem && core.getMem.hasBank, s"core ${core.getPkg}: task signature requires explicit mem.bank")
    val bytes = new java.io.ByteArrayOutputStream()
    def text(value: String): Unit = {
      bytes.write(value.getBytes(java.nio.charset.StandardCharsets.UTF_8)); bytes.write(0)
    }
    def integer(value: Long, width: Int = 8): Unit =
      (0 until width).foreach(i => bytes.write(((value >>> (8 * i)) & 255).toInt))
    text(core.getPkg)
    val bank  = core.getMem.getBank
    Seq(bank.getNum, bank.getWidth, bank.getEntries).foreach(value => integer(value.toLong))
    core.getBalldomain.getIsaList.asScala.sortBy(_.getFunct7).foreach { entry =>
      text(entry.getMnemonic); integer(entry.getFunct7.toLong, 4)
    }
    core.getBalldomain.getMappingsList.asScala.sortBy(_.getBallClass.split('.').last).foreach { mapping =>
      text(mapping.getBallClass.split('.').last)
      integer(mapping.getInBw.toLong); integer(mapping.getOutBw.toLong)
      mapping.getBallParamsMap.asScala.toSeq.sortBy(_._1).foreach { case (name, value) => text(name); text(value) }
    }
    if (core.getCpu.getKind == "ant") {
      val a      = core.getCpu.getAnt
      val shared = tss.get
      text("ant")
      Seq(
        a.getCodeBytes.toLong,
        a.getTls.getBase,
        a.getTls.getBytes.toLong,
        a.getTls.getDataBits.toLong,
        a.getTaskBits.toLong,
        shared.base.toLong,
        shared.bytes.toLong,
        shared.dataBits.toLong,
        sharedMemory.getBankEntries.toLong,
        sharedMemory.getEntries.toLong
      ).foreach(v => integer(v))
    }
    val mask  = (BigInt(1) << 64) - 1
    bytes.toByteArray.foldLeft(BigInt("cbf29ce484222325", 16)) { (hash, value) =>
      ((hash ^ BigInt(value & 255)) * BigInt("100000001b3", 16)) & mask
    }
  }

  def parseBallDomain(core: CoreInstance, repo: Path): BallDomainParam = {
    val domain   = core.getBalldomain
    val mappings = domain.getMappingsList.asScala.map { m =>
      val params = m.getBallParamsMap.asScala.toMap
      require(m.getInBw > 0 && m.getOutBw > 0, s"Ball ${m.getBallName}: BBus widths must be positive")
      val config = Some(repoFile(repo, m.getConfigPath, s"Ball ${m.getBallName} config"))

      BallIdMapping(
        ballId = m.getBallId,
        ballName = m.getBallName,
        ballClass = m.getBallClass,
        config = config,
        inBW = m.getInBw,
        outBW = m.getOutBw,
        mmioReadBW = m.getMmioReadBw,
        mmioWriteBW = m.getMmioWriteBw,
        ballParams = params
      )
    }.toSeq
    val isa      = domain.getIsaList.asScala.map { e =>
      BallISAEntry(mnemonic = e.getMnemonic, funct7 = e.getFunct7, bid = e.getBid)
    }.toSeq
    BallDomainParam(ballNum = domain.getBallNum, ballIdMappings = mappings, ballISA = isa)
  }

  def parseMemDomain(
    mem:              MemDomainConfig,
    shared:           SharedMemConfig,
    sharedBankNum:    Int,
    virtualBankCount: Int,
    nCores:           Int,
    computeCoreIds:   Seq[Int]
  ): MemDomainParam = {
    val bank = mem.getBank
    val dma  = mem.getDma
    val tlb  = mem.getTlb
    val tma  = mem.getTma
    val mmio = mem.getMmio
    MemDomainParam(
      bankNum = bank.getNum,
      bankWidth = bank.getWidth,
      bankEntries = bank.getEntries,
      bankMaskLen = bank.getMaskLen,
      virtualBankCount = virtualBankCount,
      sharedEnable = shared.getEnable,
      sharedEntries = shared.getEntries,
      sharedBankEntries = shared.getBankEntries,
      sharedBankNum = sharedBankNum,
      sharedInputChannels = shared.getInputChannels,
      sharedDefaultGroupCount = shared.getDefaultGroupCount,
      nCores = nCores,
      computeCoreIds = computeCoreIds,
      tlb_size = tlb.getSize,
      dma_n_xacts = dma.getNXacts,
      dma_burst_maxbytes = dma.getBurstMaxBytes,
      bankChannel = bank.getChannel,
      max_in_flight_mem_reqs = dma.getMaxInFlightMemReqs,
      dma_buswidth = dma.getBusWidth,
      memAddrLen = mem.getMem.getAddrLen,
      tmaReadChannel = tma.getReadChannel,
      tmaWriteChannel = tma.getWriteChannel,
      mmioEnable = mmio.getEnable,
      mmioBankNum = mmio.getBankNum,
      mmioBankEntries = mmio.getBankEntries,
      mmioBankWidth = mmio.getBankWidth,
      mmioReadWidth = mmio.getReadWidth
    )
  }

  def parseFrontend(frontend: FrontendConfig): FrontendParam =
    FrontendParam(
      rob_entries = frontend.getRobEntries,
      rs_out_of_order_response = frontend.getRsOutOfOrderResponse,
      bank_id_len = frontend.getBankIdLen,
      vbank_id_upper_bound = frontend.getVbankIdUpperBound,
      shared_bank_id_base = frontend.getSharedBankIdBase,
      iter_len = frontend.getIterLen,
      sub_rob_enable = frontend.getSubRobEnable,
      sub_rob_depth = frontend.getSubRobDepth
    )

  def parseRvv(rvv: RvvConfig): RvvParam =
    RvvParam(
      enable = rvv.getEnable,
      laneNumber = rvv.getLaneNumber,
      vLen = rvv.getVLen,
      eLen = rvv.getELen,
      iBufWords = rvv.getIBufWords,
      memoryPorts = rvv.getMemoryPorts
    )

  def parseTileParam(param: TileParamConfig): TileParam =
    TileParam(
      coreDataBytes = param.getCoreDataBytes,
      xLen = param.getXLen,
      vaddrBits = param.getVaddrBits,
      paddrBits = param.getPaddrBits,
      pgIdxBits = param.getPgIdxBits,
      pgLevels = param.getPgLevels,
      nPMPs = param.getNPmps
    )

  def parseRocketCpu(rocket: RocketCpuConfig): RocketCpuParam = {
    val mulDiv = rocket.getMulDiv
    val fpu    = rocket.getFpu
    val dcache = rocket.getDcache
    val icache = rocket.getIcache
    val btb    = rocket.getBtb
    RocketCpuParam(
      mtvecInitEnable = rocket.getMtvecInitEnable,
      useUser = rocket.getUseUser,
      useSupervisor = rocket.getUseSupervisor,
      useHypervisor = rocket.getUseHypervisor,
      useDebug = rocket.getUseDebug,
      useAtomics = rocket.getUseAtomics,
      useAtomicsOnlyForIO = rocket.getUseAtomicsOnlyForIO,
      useCompressed = rocket.getUseCompressed,
      useRVE = rocket.getUseRVE,
      useConditionalZero = rocket.getUseConditionalZero,
      nLocalInterrupts = rocket.getNLocalInterrupts,
      useNMI = rocket.getUseNMI,
      nBreakpoints = rocket.getNBreakpoints,
      useBPWatch = rocket.getUseBPWatch,
      mcontextWidth = rocket.getMcontextWidth,
      scontextWidth = rocket.getScontextWidth,
      nPerfCounters = rocket.getNPerfCounters,
      haveBasicCounters = rocket.getHaveBasicCounters,
      misaWritable = rocket.getMisaWritable,
      mtvecInit = BigInt(java.lang.Long.toUnsignedString(rocket.getMtvecInit)),
      mtvecWritable = rocket.getMtvecWritable,
      fastLoadWord = rocket.getFastLoadWord,
      fastLoadByte = rocket.getFastLoadByte,
      branchPredictionModeCSR = rocket.getBranchPredictionModeCSR,
      clockGate = rocket.getClockGate,
      mvendorid = rocket.getMvendorid,
      mimpid = rocket.getMimpid,
      haveCease = rocket.getHaveCease,
      haveSimTimeout = rocket.getHaveSimTimeout,
      useVM = rocket.getUseVm,
      useZba = rocket.getUseZba,
      useZbb = rocket.getUseZbb,
      useZbs = rocket.getUseZbs,
      haveCFlush = rocket.getHaveCFlush,
      mulDiv = MulDivParam(
        divUnroll = mulDiv.getDivUnroll,
        divEarlyOutGranularity = mulDiv.getDivEarlyOutGranularity,
        enable = mulDiv.getEnable,
        mulUnroll = mulDiv.getMulUnroll,
        mulEarlyOut = mulDiv.getMulEarlyOut,
        divEarlyOut = mulDiv.getDivEarlyOut
      ),
      fpu = FPUParam(
        divSqrt = fpu.getDivSqrt,
        sfmaLatency = fpu.getSfmaLatency,
        dfmaLatency = fpu.getDfmaLatency,
        fpmuLatency = fpu.getFpmuLatency,
        ifpuLatency = fpu.getIfpuLatency,
        enable = fpu.getEnable,
        minFLen = fpu.getMinFLen,
        fLen = fpu.getFLen
      ),
      dcache = DCacheParam(
        clockGate = dcache.getClockGate,
        nSets = dcache.getNSets,
        nWays = dcache.getNWays,
        nMSHRs = dcache.getNMshrs
      ),
      icache = ICacheParam(
        prefetch = icache.getPrefetch,
        nSets = icache.getNSets,
        nWays = icache.getNWays
      ),
      btb = BTBParam(
        nMatchBits = btb.getNMatchBits,
        nPages = btb.getNPages,
        updatesOutOfOrder = btb.getUpdatesOutOfOrder,
        bhtEnable = btb.getBhtEnable,
        bhtEntries = btb.getBhtEntries,
        bhtCounterLength = btb.getBhtCounterLength,
        bhtHistoryLength = btb.getBhtHistoryLength,
        bhtHistoryBits = btb.getBhtHistoryBits,
        enable = btb.getEnable,
        nEntries = btb.getNEntries,
        nRAS = btb.getNRas
      )
    )
  }

  def parseBoomCpu(boom: BoomCpuConfig): BoomCpuParam = {
    val dcache = boom.getDcache
    val icache = boom.getIcache
    BoomCpuParam(
      fetchWidth = boom.getFetchWidth,
      decodeWidth = boom.getDecodeWidth,
      numRobEntries = boom.getNumRobEntries,
      dcache = BoomDCacheParam(
        nSets = dcache.getNSets,
        nWays = dcache.getNWays,
        nMSHRs = dcache.getNMshrs
      ),
      icache = BoomICacheParam(
        nSets = icache.getNSets,
        nWays = icache.getNWays
      )
    )
  }

}
