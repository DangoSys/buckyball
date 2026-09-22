package framework.system.configloader

import buckyball.config.{
  BoomCpuConfig,
  Chip,
  CoreInstance,
  CpuConfig,
  FrontendConfig,
  GpDomainConfig,
  MemDomainConfig,
  RocketCpuConfig,
  SharedMemConfig,
  TileParamConfig,
  TilePlacement
}
import java.nio.file.{Files, Path, Paths}
import framework.balldomain.configs.{BallDomainParam, BallISAEntry, BallIdMapping}
import framework.frontend.configs.FrontendParam
import framework.gpdomain.configs.GpDomainParam
import framework.memdomain.configs.MemDomainParam
import framework.system.core.boom.configs.{BoomCpuParam, BoomDCacheParam, BoomICacheParam}
import framework.system.core.rocket.configs._
import framework.system.tile.PrivateDCacheParams
import framework.system.tile.configs.TileParam
import framework.top.GlobalConfig
import scala.jdk.CollectionConverters._

/** Load ExampleTopology from chip.pb. */
object ChipLoader {

  def load(pbPath: String): ExampleTopology = {
    val path  = Paths.get(pbPath)
    if (!Files.isRegularFile(path)) {
      throw new RuntimeException(s"chip.pb does not exist: $pbPath")
    }
    val repo  = repoRoot(path)
    val chip  = Chip.parseFrom(Files.readAllBytes(path))
    val cores = chip.getCoresList.asScala.toSeq
    if (cores.isEmpty) {
      throw new RuntimeException(s"chip.pb has no cores: $pbPath")
    }
    val tiles = chip.getTilesList.asScala.map(parseTile(_, cores, repo)).toSeq
    require(
      tiles.size == chip.getNTiles,
      s"chip.pb declares top.nTiles=${chip.getNTiles} but defines ${tiles.size} tile(s) in $pbPath"
    )
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
    val indices      = tile.getCoreIndicesList.asScala.map(_.toInt).toSeq
    val nCores       = indices.size
    val hasBuckyball = indices.exists(i => cores(i).getBalldomain.getBallNum > 0)
    val tileParam    = parseTileParam(tile.getParam)

    val shared           = tile.getSharedMem
    val virtualBankCount = tile.getVirtualBankCount
    require(indices.nonEmpty, s"tile ${tile.getPath}: core_indices must not be empty")
    if (hasBuckyball) {
      require(virtualBankCount > 0, s"tile ${tile.getPath}: virtual_bank_count must be > 0")
    }
    val sharedBankNum    =
      if (shared.getEnable) {
        val firstBank = cores(indices.head).getMem.getBank
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
    val privateDCache    =
      if (!tile.hasPrivateDcache || !tile.getPrivateDcache.getEnable) None
      else {
        val dcache = tile.getPrivateDcache
        val ways   = dcache.getWays
        val sets   = (dcache.getCapacityKb * 1024) / (64 * ways)
        Some(PrivateDCacheParams(
          ways = ways,
          sets = sets,
          writeBytes = dcache.getWriteBytes,
          portFactor = dcache.getPortFactor,
          memCycles = dcache.getMemCycles
        ))
      }

    val tileCores = indices.map { idx =>
      parseCore(cores(idx), tileParam, shared, sharedBankNum, virtualBankCount, nCores, repo)
    }
    TileTopology(tileParam, tileCores, privateDCache)
  }

  private def parseCore(
    core:             CoreInstance,
    tile:             TileParam,
    shared:           SharedMemConfig,
    sharedBankNum:    Int,
    virtualBankCount: Int,
    nCores:           Int,
    repo:             Path
  ): TileCore = {
    require(core.hasCpu, s"core ${core.getPkg}: missing cpu config")
    val cpu  = core.getCpu
    val kind = cpu.getKind
    kind match {
      case "rocket" => parseRocketCoreSlot(core, tile, shared, sharedBankNum, virtualBankCount, nCores, repo)
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
    require(core.hasGpDomain, s"core ${core.getPkg} missing gpdomain config")
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
      ballDomain = parseBallDomain(core, repo),
      frontend = parseFrontend(core.getFrontend),
      gpDomain = parseGpDomain(core.getGpDomain),
      tile = tile,
      memDomain = parseMemDomain(core.getMem, shared, sharedBankNum, virtualBankCount, nCores)
    )
    RocketTileCore(rocket, Some(buckyball))
  }

  private def parseBallDomain(core: CoreInstance, repo: Path): BallDomainParam = {
    val domain   = core.getBalldomain
    val mappings = domain.getMappingsList.asScala.map { m =>
      BallIdMapping(
        ballId = m.getBallId,
        ballName = m.getBallName,
        ballClass = m.getBallClass,
        config = Some(repoFile(repo, m.getConfigPath, s"Ball ${m.getBallName} config")),
        inBW = m.getInBw,
        outBW = m.getOutBw,
        mmioReadBW = m.getMmioReadBw,
        mmioWriteBW = m.getMmioWriteBw
      )
    }.toSeq
    val isa      = domain.getIsaList.asScala.map { e =>
      BallISAEntry(mnemonic = e.getMnemonic, funct7 = e.getFunct7, bid = e.getBid)
    }.toSeq
    BallDomainParam(ballNum = domain.getBallNum, ballIdMappings = mappings, ballISA = isa)
  }

  private def parseMemDomain(
    mem:              MemDomainConfig,
    shared:           SharedMemConfig,
    sharedBankNum:    Int,
    virtualBankCount: Int,
    nCores:           Int
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

  private def parseGpDomain(gp: GpDomainConfig): GpDomainParam =
    GpDomainParam(
      laneNumber = gp.getLaneNumber,
      chainingSize = gp.getChainingSize,
      vLen = gp.getVLen,
      dLen = gp.getDLen,
      eLen = gp.getELen,
      laneScale = gp.getLaneScale
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
