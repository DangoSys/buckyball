package sims.p2e

import chisel3._
import _root_.circt.stage.ChiselStage
import org.chipsalliance.cde.config.Config

import freechips.rocketchip.devices.tilelink.{BootROMLocated, BootROMParams}
import freechips.rocketchip.subsystem.{InSubsystem, WithCustomMemPort}
import sims.scu.WithSCU

class WithP2EBootROM
    extends Config((site, here, up) => {
      case BootROMLocated(InSubsystem) => Some(BootROMParams(
          contentFileName = "src/main/resources/bootrom/bootrom.rv64.img"
        ))
    })

/**
 * Linux BootROM for P2E: jumps to OpenSBI fw_payload at 0x80000000.
 * Use this instead of WithP2EBootROM when running Linux.
 */
class WithLinuxBootROM
    extends Config((site, here, up) => {
      case BootROMLocated(InSubsystem) => Some(BootROMParams(
          contentFileName = "src/main/resources/linux/bootrom.rv64.img"
        ))
    })

class WithP2EDDR4MemPort
    extends Config(
      new WithCustomMemPort(
        base_addr = BigInt("80000000", 16),
        base_size = BigInt("400000000", 16),
        data_width = 256,
        id_bits = 11,
        maxXferBytes = 256
      )
    )

// =============================================================================
// P2EBaseConfig: P2E platform-specific fragments only.
// The full base (clocking, buses, BootROM, etc.) comes from BuckyballBaseConfig
// which is included in the example SoC config (e.g. BuckyballToyConfig).
//
// P2E adds:
//   - WithP2EHarness    : P2E harness binders (DDR4 wiring, etc.)
//   - WithSCU        : per-tile UART/exit via DPI-C (intercepted in BBTile)
//   - WithP2EDDR4MemPort: DDR4 memory port @ 0x80000000, 16 GiB
//   - WithP2EBootROM    : P2E bootrom image
// =============================================================================
class P2EBaseConfig(maxHarts: Int = 64)
    extends Config(
      new WithP2EHarness ++
        new WithSCU(maxHarts = maxHarts) ++
        new WithP2EDDR4MemPort ++
        new WithP2EBootROM
    )

class P2EToyConfig
    extends Config(
      new P2EBaseConfig ++
        new examples.toy.BuckyballToyConfig
    )

/**
 * Linux variant of P2EToyConfig.
 * Uses linux/bootrom.rv64.img which jumps to OpenSBI fw_payload at 0x80000000.
 * Pair with OpenSBI fw_payload built by `bbdev kernel --build`.
 */
class P2EToyLinuxConfig
    extends Config(
      new WithLinuxBootROM ++
        new P2EBaseConfig ++
        new examples.toy.BuckyballToyConfig
    )

//===----------------------------------------------------------------------===//
// Gemmini P2E configs
//===----------------------------------------------------------------------===//
/**
 * P2E Gemmini config without Debug module.
 * Uses the same Gemmini + Rocket configuration as chipyard.GemminiRocketConfig
 * but replaces AbstractConfig with BuckyballBaseConfig to avoid Debug/UART/SerialTL.
 */
class P2EGemminiConfig
    extends Config(
      new P2EBaseConfig ++
        new gemmini.DefaultGemminiConfig ++
        new freechips.rocketchip.rocket.WithNHugeCores(1) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )

/**
 * Linux variant of P2EGemminiConfig.
 * Uses linux/bootrom.rv64.img which jumps to OpenSBI fw_payload at 0x80000000.
 * Pair with OpenSBI fw_payload built by `bbdev kernel --build`.
 */
class P2EGemminiLinuxConfig
    extends Config(
      new WithLinuxBootROM ++
        new P2EBaseConfig ++
        new gemmini.DefaultGemminiConfig ++
        new freechips.rocketchip.rocket.WithNHugeCores(1) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )

//===----------------------------------------------------------------------===//
// Goban P2E configs
//===----------------------------------------------------------------------===//
class BuckyballGoban2CoreP2EConfig
    extends Config(
      new P2EBaseConfig ++
        new examples.goban.BuckyballGoban2CoreConfig
    )

class BuckyballGoban4CoreP2EConfig
    extends Config(
      new P2EBaseConfig ++
        new examples.goban.BuckyballGoban4CoreConfig
    )

class BuckyballGoban8CoreP2EConfig
    extends Config(
      new P2EBaseConfig ++
        new examples.goban.BuckyballGoban8CoreConfig
    )

class BuckyballGoban64CoreP2EConfig
    extends Config(
      new P2EBaseConfig ++
        new examples.goban.BuckyballGoban64CoreConfig
    )

class BuckyballGoban24Tile16CoreP2EConfig
    extends Config(
      new P2EBaseConfig ++
        new examples.goban.BuckyballGoban24Tile16CoreConfig
    )

class BuckyballGoban2Tile4CoreP2EConfig
    extends Config(
      new P2EBaseConfig(maxHarts = 8) ++
        new examples.goban.BuckyballGoban2Tile4CoreConfig
    )

/**
 * Linux variant of the 2-tile Goban config.
 * Uses linux/bootrom.rv64.img which jumps to OpenSBI fw_payload at 0x80000000.
 */
class BuckyballGoban2Tile4CoreLinuxP2EConfig
    extends Config(
      new WithLinuxBootROM ++
        new P2EBaseConfig(maxHarts = 8) ++
        new examples.goban.BuckyballGoban2Tile4CoreConfig
    )

class BuckyballGoban64Tile4CoreP2EConfig
    extends Config(
      new P2EBaseConfig(maxHarts = 256) ++
        new examples.goban.BuckyballGoban64Tile4CoreConfig
    )

/**
 * Linux variant of the 64-tile Goban config.
 * Uses linux/bootrom.rv64.img which jumps to OpenSBI fw_payload at 0x80000000.
 */
class BuckyballGoban64Tile4CoreLinuxP2EConfig
    extends Config(
      new WithLinuxBootROM ++
        new P2EBaseConfig(maxHarts = 256) ++
        new examples.goban.BuckyballGoban64Tile4CoreConfig
    )

class BuckyballGobanConfig1LinuxP2EConfig
    extends Config(
      new WithLinuxBootROM ++
        new P2EBaseConfig(maxHarts = 128) ++
        new examples.goban.BuckyballGobanConfig1Config
    )

class BuckyballGobanConfig2LinuxP2EConfig
    extends Config(
      new WithLinuxBootROM ++
        new P2EBaseConfig(maxHarts = 192) ++
        new examples.goban.BuckyballGobanConfig2Config
    )

class BuckyballGobanConfig3LinuxP2EConfig
    extends Config(
      new WithLinuxBootROM ++
        new P2EBaseConfig(maxHarts = 128) ++
        new examples.goban.BuckyballGobanConfig3Config
    )

class BuckyballGobanConfig4LinuxP2EConfig
    extends Config(
      new WithLinuxBootROM ++
        new P2EBaseConfig(maxHarts = 256) ++
        new examples.goban.BuckyballGobanConfig4Config
    )

//===----------------------------------------------------------------------===//

object Elaborate extends App {
  if (args.isEmpty) {
    println("Usage: Elaborate <full.config.ClassName> [firtool-opts...]")
    println("Example: Elaborate sims.p2e.P2EToyConfig")
    sys.exit(1)
  }

  val configClassName = args(0)
  println(s"Elaborating P2EHarness with config: $configClassName")

  val config: Config =
    try {
      val configClass = Class.forName(configClassName)
      configClass.getDeclaredConstructor().newInstance().asInstanceOf[Config]
    } catch {
      case e: ClassNotFoundException =>
        println(s"Error: Config class not found: $configClassName")
        sys.exit(1)
      case e: Exception              =>
        println(s"Error loading config class: ${e.getMessage}")
        e.printStackTrace()
        sys.exit(1)
    }

  ChiselStage.emitSystemVerilogFile(
    new P2EHarness()(config.toInstance),
    firtoolOpts = args.drop(1),
    args = Array.empty
  )
}
