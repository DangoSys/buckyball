package sims.p2e

import chisel3._
import _root_.circt.stage.ChiselStage
import org.chipsalliance.cde.config.Config
import freechips.rocketchip.devices.tilelink.{BootROMLocated, BootROMParams}
import freechips.rocketchip.subsystem.{InSubsystem, WithCustomMemPort}
import sims.scu.WithSCU

class WithP2EBootROM
    extends Config((site, here, up) => {
      case BootROMLocated(InSubsystem) => Seq(BootROMParams(
          contentFileName = freechips.rocketchip.util.SystemFileName("src/main/resources/bootrom/bare/bootrom.rv64.img")
        ))
    })

/**
 * Linux BootROM for P2E: jumps to OpenSBI fw_payload at 0x80000000.
 * Use this instead of WithP2EBootROM when running Linux.
 */
class WithLinuxBootROM
    extends Config((site, here, up) => {
      case BootROMLocated(InSubsystem) => Seq(BootROMParams(
          contentFileName = freechips.rocketchip.util.SystemFileName("src/main/resources/bootrom/linux/bootrom.rv64.img")
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
//   - WithSCU           : per-tile UART/exit via DPI-C (intercepted in BBTile)
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
        new freechips.rocketchip.rocket.WithNHugeCores(1) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )

/**
 * Linux variant of P2EGemminiConfig.
 * Uses bootrom/linux/bootrom.rv64.img which jumps to OpenSBI fw_payload at 0x80000000.
 * Pair with OpenSBI fw_payload built by `bbdev kernel --build`.
 */
class P2EGemminiLinuxConfig
    extends Config(
      new WithLinuxBootROM ++
        new P2EBaseConfig ++
        new freechips.rocketchip.rocket.WithNHugeCores(1) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
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

  val firtoolOpts = args.drop(1)

  ChiselStage.emitSystemVerilogFile(
    new P2EHarness()(config.toInstance),
    firtoolOpts = firtoolOpts,
    args = Array.empty
  )
  ChiselStage.emitSystemVerilogFile(
    new P2ETop,
    firtoolOpts = firtoolOpts,
    args = Array.empty
  )
}
