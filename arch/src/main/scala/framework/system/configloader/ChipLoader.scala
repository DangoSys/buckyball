package framework.system.configloader

import buckyball.config.{
  Chip,
  CoreInstance,
  CoreParamConfig,
  FrontendConfig,
  GpDomainConfig,
  MemDomainConfig,
  RocketCoreConfig,
  SharedMemConfig,
  TilePlacement
}
import java.nio.file.{Files, Path, Paths}
import framework.balldomain.configs.{BallDomainParam, BallISAEntry, BallIdMapping}
import framework.frontend.configs.FrontendParam
import framework.gpdomain.configs.GpDomainParam
import framework.memdomain.configs.MemDomainParam
import framework.system.core.configs._
import framework.system.tile.PrivateDCacheParams
import framework.top.GlobalConfig
import framework.top.configs.TopConfig
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
    val indices           = tile.getCoreIndicesList.asScala.map(_.toInt).toSeq
    val nCores            = indices.size
    val hasBuckyball      = indices.exists(i => cores(i).getBalldomain.getBallNum > 0)
    val memBallChannelNum =
      if (hasBuckyball) tile.getMemBallChannelNum
      else 0

    val shared        = tile.getSharedMem
    val privateDCache =
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

    val coreEntries    = indices.map { idx =>
      parseCore(cores(idx), shared, memBallChannelNum, nCores, repo)
    }
    val buckyballCores = coreEntries.map(_._1).map(_.map { cfg =>
      cfg.copy(top = TopConfig(memBallChannelNum = memBallChannelNum, nCores = nCores))
    })
    val rocketCores    = coreEntries.map {
      case (_, rocket) => rocket
    }
    TileTopology(buckyballCores, privateDCache, rocketCores)
  }

  private def parseCore(
    core:              CoreInstance,
    shared:            SharedMemConfig,
    memBallChannelNum: Int,
    nCores:            Int,
    repo:              Path
  ): (Option[GlobalConfig], RocketCoreParam) = {
    val rocket = parseRocketCore(core.getRocketCore)
    val domain = core.getBalldomain
    if (domain.getBallNum == 0) {
      return (None, rocket)
    }
    require(core.hasFrontend, s"core ${core.getPkg} missing frontend config")
    require(core.hasGpDomain, s"core ${core.getPkg} missing gpdomain config")
    require(core.hasCore, s"core ${core.getPkg} missing core config")

    val buckyball = GlobalConfig().copy(
      ballDomain = parseBallDomain(core, repo),
      frontend = parseFrontend(core.getFrontend),
      gpDomain = parseGpDomain(core.getGpDomain),
      core = parseCoreParam(core.getCore),
      memDomain = parseMemDomain(core.getMem, shared),
      rocketCore = rocket,
      top = TopConfig(memBallChannelNum = memBallChannelNum, nCores = nCores)
    )
    (Some(buckyball), rocket)
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

  private def parseMemDomain(mem: MemDomainConfig, shared: SharedMemConfig): MemDomainParam = {
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
      sharedEnable = shared.getEnable,
      sharedEntries = shared.getEntries,
      sharedInputChannels = shared.getInputChannels,
      sharedDefaultGroupCount = shared.getDefaultGroupCount,
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

  private def parseCoreParam(core: CoreParamConfig): CoreParam =
    CoreParam(
      coreDataBytes = core.getCoreDataBytes,
      xLen = core.getXLen,
      vaddrBits = core.getVaddrBits,
      paddrBits = core.getPaddrBits,
      pgIdxBits = core.getPgIdxBits,
      nPMPs = core.getNPmps
    )

  private def parseRocketCore(rocket: RocketCoreConfig): RocketCoreParam = {
    val mulDiv = rocket.getMulDiv
    val fpu    = rocket.getFpu
    val dcache = rocket.getDcache
    val icache = rocket.getIcache
    val btb    = rocket.getBtb
    RocketCoreParam(
      xLen = rocket.getXLen,
      pgLevels = rocket.getPgLevels,
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

}
