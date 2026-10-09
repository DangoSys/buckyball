package fpga.arty

import java.nio.file.Paths
import org.chipsalliance.cde.config.Config
import freechips.rocketchip.devices.debug.DebugModuleKey
import freechips.rocketchip.devices.tilelink.{BootROMLocated, BootROMParams}
import freechips.rocketchip.subsystem._
import freechips.rocketchip.util.SystemFileName
import framework.balldomain.configs.{BallDomainParam, BallISAEntry, BallIdMapping}
import framework.frontend.configs.FrontendParam
import framework.gpdomain.configs.GpDomainParam
import framework.memdomain.configs.MemDomainParam
import framework.system.core.rocket.configs._
import framework.system.tile.{BBTileAttachParams, WithBBTile}
import framework.system.tile.configs.TileParam
import framework.top.GlobalConfig
import framework.top.configs.SimParam

// FPGA-specific configuration only. Upstream CPU, accelerator, DMA, and
// operator sources remain unchanged under upstream/.
object PebbleA7Parameters {

  val tile = TileParam(
    coreDataBytes = 8,
    xLen = 64,
    vaddrBits = 39,
    paddrBits = 32,
    pgIdxBits = 12,
    pgLevels = 3,
    nPMPs = 8
  )

  val cpu = RocketCpuParam(
    useVM = false,
    useZba = false,
    useZbb = false,
    useZbs = false,
    mulDiv = MulDivParam(enable = true, mulUnroll = 1),
    fpu = FPUParam(enable = false),
    dcache = DCacheParam(nSets = 32, nWays = 2, nMSHRs = 0),
    icache = ICacheParam(nSets = 32, nWays = 2),
    btb = BTBParam(enable = false)
  )

  private def source(path: String): String = Paths.get("upstream", path).toAbsolutePath.toString

  val accelerator = GlobalConfig(
    memDomain = MemDomainParam(
      bankNum = 8,
      bankWidth = 128,
      bankEntries = 512,
      bankMaskLen = 16,
      virtualBankCount = 8,
      sharedEnable = false,
      sharedEntries = 512,
      sharedBankNum = 8,
      sharedInputChannels = 2,
      sharedDefaultGroupCount = 1,
      nCores = 1,
      tlb_size = 4,
      dma_n_xacts = 2,
      dma_burst_maxbytes = 64,
      bankChannel = 5,
      max_in_flight_mem_reqs = 4,
      dma_buswidth = 128,
      memAddrLen = 32,
      tmaReadChannel = 2,
      tmaWriteChannel = 1,
      mmioEnable = false,
      mmioBankNum = 1,
      mmioBankEntries = 16,
      mmioBankWidth = 8,
      mmioReadWidth = 8
    ),
    frontend = FrontendParam(
      rob_entries = 8,
      rs_out_of_order_response = false,
      bank_id_len = 10,
      vbank_id_upper_bound = 7,
      shared_bank_id_base = 512,
      iter_len = 16,
      sub_rob_enable = false,
      sub_rob_depth = 4
    ),
    gpDomain = GpDomainParam(
      laneNumber = 1,
      chainingSize = 2,
      vLen = 128,
      dLen = 128,
      eLen = 32,
      laneScale = 1
    ),
    ballDomain = BallDomainParam(
      ballNum = 3,
      ballIdMappings = Seq(
        BallIdMapping(
          0,
          "SMatMulBall",
          "examples.balls.smatmul.SMatMulBall",
          Some(source("examples/cores/pebble/configs/balldomains/balls/smatmul/8x16.toml")),
          2,
          1
        ),
        BallIdMapping(
          1,
          "MatAddBall",
          "examples.balls.matadd.MatAddBall",
          Some(source("examples/balls/matadd/configs/default.toml")),
          2,
          1
        ),
        BallIdMapping(
          2,
          "ReluBall",
          "examples.balls.relu.ReluBall",
          Some(source("examples/balls/relu/configs/default.toml")),
          1,
          1
        )
      ),
      ballISA = Seq(
        BallISAEntry("SMATMUL_OS", 65, 0),
        BallISAEntry("SMATMUL_BIAS", 17, 0),
        BallISAEntry("MATADD", 72, 1),
        BallISAEntry("RELU", 50, 2)
      )
    ),
    tile = tile,
    sim = SimParam(diffTest = false)
  )

}

// The AXI memory port terminates in FPGA BRAM; the MMIO port terminates in
// real UART/status peripherals. No simulation DRAM or DPI console is used.
class WithArtyA7Ports
    extends Config((site, here, up) => {
      case ExtMem                      => Some(MemoryPortParams(
          master = MasterPortParams(BigInt("80000000", 16), 128 * 1024, 16, 4, 64),
          nMemoryChannels = 1
        ))
      case ExtBus                      => Some(MasterPortParams(BigInt("60000000", 16), 4096, 8, 2, 8))
      case ExtIn                       => None
      case DebugModuleKey              => None
      case BootROMLocated(InSubsystem) => Seq(BootROMParams(
          hang = BigInt("10000", 16),
          appendDTB = false,
          contentFileName = SystemFileName("firmware/bootrom.bin")
        ))
      case SystemBusKey                => up(SystemBusKey, site).copy(beatBytes = 16)
      case MemoryBusKey                => up(MemoryBusKey, site).copy(beatBytes = 16)
      case TilesLocated(InSubsystem)   => up(TilesLocated(InSubsystem), site).map {
          case attachment: BBTileAttachParams =>
            val tile = attachment.tileParams
            attachment.copy(tileParams = tile.copy(
              core = tile.core.copy(clockGate = false, haveSimTimeout = false),
              rocketCorePerCore = tile.rocketCorePerCore.map(_.copy(clockGate = false, haveSimTimeout = false)),
              dcache = tile.dcache.map(_.copy(clockGate = false))
            ))
          case other => other
        }
    })

class PebbleA7Config
    extends Config(
      new WithArtyA7Ports ++
        new WithBBTile(
          tileParam = PebbleA7Parameters.tile,
          buckyballConfig = PebbleA7Parameters.accelerator,
          rocketCpuPerCore = Some(Seq(PebbleA7Parameters.cpu))
        ) ++
        new WithNExtTopInterrupts(0) ++
        new WithCoherentBusTopology ++
        new freechips.rocketchip.system.BaseConfig
    )
