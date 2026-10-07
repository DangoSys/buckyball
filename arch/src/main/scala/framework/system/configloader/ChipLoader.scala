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

  private def repoRoot(pb: Path): Path = {
    val abs    = pb.toAbsolutePath.normalize
    val s      = abs.toString
    val marker = "/examples/chips/"
    val i      = s.lastIndexOf(marker)
    if (i < 0) {
      throw new RuntimeException(s"chip.pb is not under examples/chips: $pb")
    }
    Paths.get(s.substring(0, i))
  }

  private def repoFile(repo: Path, rel: String, what: String): String = {
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

  private def parseTile(tile: TilePlacement, cores: Seq[CoreInstance], repo: Path): TileTopology = {
    val indices        = tile.getCoreIndicesList.asScala.map(_.toInt).toSeq
    val nCores         = indices.size
    val computeCoreIds =
      indices.zipWithIndex.collect { case (index, slot) if cores(index).getBalldomain.getBallNum > 0 => slot }
    val hasBuckyball   = computeCoreIds.nonEmpty
    val tileParam      = parseTileParam(tile.getParam)

    val shared           = tile.getSharedMem
    val virtualBankCount = tile.getVirtualBankCount
    require(indices.nonEmpty, s"tile ${tile.getPath}: core_indices must not be empty")
    if (hasBuckyball) {
      require(virtualBankCount > 0, s"tile ${tile.getPath}: virtual_bank_count must be > 0")
    }
    val sharedBankNum    =
      if (shared.getEnable) {
        val firstBank = cores(indices(computeCoreIds.head)).getMem.getBank
        indices.iterator.map(idx => cores(idx)).filter(_.getBalldomain.getBallNum > 0).foreach { core =>
          require(
            core.getMem.getBank.getWidth == firstBank.getWidth,
            s"tile ${tile.getPath}: all Buckyball cores must use shared bank width ${firstBank.getWidth}"
          )
        }
        require(shared.getEntries > 0, s"tile ${tile.getPath}: shared entries must be > 0")
        require(
          shared.getEntries % firstBank.getEntries == 0,
          s"tile ${tile.getPath}: shared entries ${shared.getEntries} must be divisible by slot-0 bank entries ${firstBank.getEntries}"
        )
        shared.getEntries / firstBank.getEntries
      } else 0
    val tileCores        = indices.map { idx =>
      parseCore(cores(idx), tileParam, shared, sharedBankNum, virtualBankCount, nCores, computeCoreIds, repo)
    }
    val hartIds          = indices.map { idx =>
      require(cores(idx).hasHartId, s"core ${cores(idx).getPkg}: missing explicit hart_id")
      cores(idx).getHartId
    }
    val main             = tile.getKind == TileKind.TILE_KIND_MAIN
    require(main || tile.hasControllerCoreIndex, s"tile ${tile.getPath}: a compute tile needs controller_core_index")
    val controller       =
      if (tile.hasControllerCoreIndex) {
        val local = indices.indexOf(tile.getControllerCoreIndex)
        require(
          local == 0 && indices.size > 1,
          s"tile ${tile.getPath}: controller_core_index must select local slot 0 with workers"
        )
        Some(local)
      } else None
    TileTopology(
      tileParam,
      main,
      tileCores,
      hartIds,
      if (controller.isDefined) indices.map(idx => coreSignature(cores(idx))) else Nil,
      controller
    )
  }

  private def parseCore(
    core:             CoreInstance,
    tile:             TileParam,
    shared:           SharedMemConfig,
    sharedBankNum:    Int,
    virtualBankCount: Int,
    nCores:           Int,
    computeCoreIds:   Seq[Int],
    repo:             Path
  ): TileCore = {
    require(core.hasCpu, s"core ${core.getPkg}: missing cpu config")
    val cpu  = core.getCpu
    val kind = cpu.getKind
    kind match {
      case "rocket" =>
        parseRocketCoreSlot(core, tile, shared, sharedBankNum, virtualBankCount, nCores, computeCoreIds, repo)
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

  private def parseRocketCoreSlot(
    core:             CoreInstance,
    tile:             TileParam,
    shared:           SharedMemConfig,
    sharedBankNum:    Int,
    virtualBankCount: Int,
    nCores:           Int,
    computeCoreIds:   Seq[Int],
    repo:             Path
  ): RocketTileCore = {
    val cpu      = core.getCpu
    if (!cpu.hasRocket) {
      throw new RuntimeException(s"core ${core.getPkg}: kind=rocket missing cpu.rocket")
    }
    val rocket   = parseRocketCpu(cpu.getRocket)
    val domain   = core.getBalldomain
    if (domain.getBallNum == 0) {
      return RocketTileCore(rocket, None)
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
      coreSignature = coreSignature(core),
      ballDomain = parseBallDomain(core, repo),
      frontend = parseFrontend(core.getFrontend),
      rvv = parseRvv(core.getRvv),
      tile = tile,
      memDomain = parseMemDomain(core.getMem, shared, sharedBankNum, virtualBankCount, nCores, computeCoreIds)
    )
    RocketTileCore(rocket, Some(buckyball))
  }

  private def coreSignature(core: CoreInstance): BigInt = {
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
    val mask  = (BigInt(1) << 64) - 1
    bytes.toByteArray.foldLeft(BigInt("cbf29ce484222325", 16)) { (hash, value) =>
      ((hash ^ BigInt(value & 255)) * BigInt("100000001b3", 16)) & mask
    }
  }

  private def parseBallDomain(core: CoreInstance, repo: Path): BallDomainParam = {
    val domain      = core.getBalldomain
    val mappings    = domain.getMappingsList.asScala.map { m =>
      val params = m.getBallParamsMap.asScala.toMap
      val config = m.getBuiltin match {
        case ""       =>
          require(m.getInBw > 0 && m.getOutBw > 0, s"Ball ${m.getBallName}: ordinary Ball must have positive BBus widths")
          Some(repoFile(repo, m.getConfigPath, s"Ball ${m.getBallName} config"))
        case "kernel" =>
          require(
            m.getBallName == "kernel" && m.getBallClass == "framework.balldomain.kernel.KernelBall",
            "builtin kernel must use the kernel name and KernelBall class"
          )
          require(
            m.getInBw == 0 && m.getOutBw == 0 && m.getMmioReadBw == 0 && m.getMmioWriteBw == 0,
            "builtin kernel must not have BBus or MMIO widths"
          )
          require(
            m.getConfigPath.isEmpty && m.getBallDir.isEmpty,
            "builtin kernel must not reference external config or ball_dir"
          )
          require(
            core.hasRvv && core.getRvv.hasEnable && core.getRvv.getEnable,
            "builtin kernel requires explicitly enabled RVV"
          )
          val rvv      = core.getRvv
          val expected = Map(
            "laneNumber"  -> rvv.getLaneNumber.toString,
            "vLen"        -> rvv.getVLen.toString,
            "eLen"        -> rvv.getELen.toString,
            "iBufWords"   -> rvv.getIBufWords.toString,
            "memoryPorts" -> rvv.getMemoryPorts.toString
          )
          require(params == expected, "builtin kernel parameters must exactly match the core RVV configuration")
          require(
            m.getBallId == domain.getBallNum - 1 &&
              domain.getMappingsList.get(domain.getMappingsCount - 1).getBallId == m.getBallId,
            "builtin kernel must be the final configured Ball"
          )
          require(
            domain.getIsaList.asScala.exists(e =>
              e.getMnemonic == "RUN_KERNEL" &&
                e.getFunct7 == 15 && e.getBid == m.getBallId
            ),
            "builtin kernel requires RUN_KERNEL funct7 15 mapped to its Ball ID"
          )
          None
        case builtin  => throw new IllegalArgumentException(s"Unknown builtin Ball: $builtin")
      }
      BallIdMapping(
        ballId = m.getBallId,
        ballName = m.getBallName,
        ballClass = m.getBallClass,
        config = config,
        inBW = m.getInBw,
        outBW = m.getOutBw,
        mmioReadBW = m.getMmioReadBw,
        mmioWriteBW = m.getMmioWriteBw,
        builtin = m.getBuiltin,
        ballParams = params
      )
    }.toSeq
    val kernelCount = mappings.count(_.builtin == "kernel")
    val rvvEnabled  = core.hasRvv && core.getRvv.hasEnable && core.getRvv.getEnable
    require(
      kernelCount == (if (rvvEnabled) 1 else 0),
      "BallDomain must contain exactly one builtin kernel when RVV is enabled"
    )
    val isa         = domain.getIsaList.asScala.map { e =>
      BallISAEntry(mnemonic = e.getMnemonic, funct7 = e.getFunct7, bid = e.getBid)
    }.toSeq
    BallDomainParam(ballNum = domain.getBallNum, ballIdMappings = mappings, ballISA = isa)
  }

  private def parseMemDomain(
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

  private def parseFrontend(frontend: FrontendConfig): FrontendParam =
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

  private def parseRvv(rvv: RvvConfig): RvvParam =
    RvvParam(
      enable = rvv.getEnable,
      laneNumber = rvv.getLaneNumber,
      vLen = rvv.getVLen,
      eLen = rvv.getELen,
      iBufWords = rvv.getIBufWords,
      memoryPorts = rvv.getMemoryPorts
    )

  private def parseTileParam(param: TileParamConfig): TileParam =
    TileParam(
      coreDataBytes = param.getCoreDataBytes,
      xLen = param.getXLen,
      vaddrBits = param.getVaddrBits,
      paddrBits = param.getPaddrBits,
      pgIdxBits = param.getPgIdxBits,
      pgLevels = param.getPgLevels,
      nPMPs = param.getNPmps
    )

  private def parseRocketCpu(rocket: RocketCpuConfig): RocketCpuParam = {
    val mulDiv = rocket.getMulDiv
    val fpu    = rocket.getFpu
    val dcache = rocket.getDcache
    val icache = rocket.getIcache
    val btb    = rocket.getBtb
    RocketCpuParam(
      useVM = rocket.getUseVm,
      useZba = rocket.getUseZba,
      useZbb = rocket.getUseZbb,
      useZbs = rocket.getUseZbs,
      haveCFlush = rocket.getHaveCFlush,
      mulDiv = MulDivParam(
        enable = mulDiv.getEnable,
        mulUnroll = mulDiv.getMulUnroll,
        mulEarlyOut = mulDiv.getMulEarlyOut,
        divEarlyOut = mulDiv.getDivEarlyOut
      ),
      fpu = FPUParam(
        enable = fpu.getEnable,
        minFLen = fpu.getMinFLen,
        fLen = fpu.getFLen
      ),
      dcache = DCacheParam(
        nSets = dcache.getNSets,
        nWays = dcache.getNWays,
        nMSHRs = dcache.getNMshrs
      ),
      icache = ICacheParam(
        nSets = icache.getNSets,
        nWays = icache.getNWays
      ),
      btb = BTBParam(
        enable = btb.getEnable,
        nEntries = btb.getNEntries,
        nRAS = btb.getNRas
      )
    )
  }

  private def parseBoomCpu(boom: BoomCpuConfig): BoomCpuParam = {
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
