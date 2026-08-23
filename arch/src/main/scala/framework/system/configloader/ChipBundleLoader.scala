package framework.system.configloader

import buckyball.config.{
  ChipBundle,
  CoreInstance,
  CoreParamConfig,
  FrontendConfig,
  GpDomainConfig,
  MemDomainConfig,
  RocketCoreConfig,
  SharedMemConfig,
  TilePlacement
}
import java.nio.file.{Files, Paths}
import framework.balldomain.configs.{BallDomainParam, BallISAEntry, BallIdMapping}
import framework.frontend.configs.FrontendParam
import framework.gpdomain.configs.GpDomainParam
import framework.memdomain.configs.MemDomainParam
import framework.system.core.configs._
import framework.system.tile.PrivateDCacheParams
import framework.top.GlobalConfig
import framework.top.configs.TopConfig
import scala.jdk.CollectionConverters._

/** Load ExampleTopology from a ChipBundle protobuf written by chip_bundle.py. */
object ChipBundleLoader {

  def load(pbPath: String): ExampleTopology = {
    val path   = Paths.get(pbPath)
    if (!Files.isRegularFile(path)) {
      throw new RuntimeException(s"ChipBundle does not exist: $pbPath")
    }
    val bundle = ChipBundle.parseFrom(Files.readAllBytes(path))
    val cores  = bundle.getCoresList.asScala.toSeq
    if (cores.isEmpty) {
      throw new RuntimeException(s"ChipBundle has no cores: $pbPath")
    }
    val tiles  = bundle.getTilesList.asScala.map(parseTile(_, cores)).toSeq
    require(
      tiles.size == bundle.getNTiles,
      s"ChipBundle declares top.nTiles=${bundle.getNTiles} but defines ${tiles.size} tile(s) in $pbPath"
    )
    ExampleTopology(tiles)
  }

  private def parseTile(tile: TilePlacement, cores: Seq[CoreInstance]): TileTopology = {
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
      parseCore(cores(idx), shared, memBallChannelNum, nCores)
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
    nCores:            Int
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
      ballDomain = parseBallDomain(core),
      frontend = parseFrontend(core.getFrontend),
      gpDomain = parseGpDomain(core.getGpDomain),
      core = parseCoreParam(core.getCore),
      memDomain = parseMemDomain(core.getMem, shared),
      rocketCore = rocket,
      top = TopConfig(memBallChannelNum = memBallChannelNum, nCores = nCores)
    )
    (Some(buckyball), rocket)
  }

  private def parseBallDomain(core: CoreInstance): BallDomainParam = {
    val domain   = core.getBalldomain
    val mappings = domain.getMappingsList.asScala.map { m =>
      BallIdMapping(
        ballId = m.getBallId,
        ballName = m.getBallName,
        ballClass = m.getBallClass,
        config = Option(m.getConfigPath).filter(_.nonEmpty),
        inBW = m.getInBw,
        outBW = m.getOutBw,
        configBaseDir = core.getBalldomainBaseDir
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
