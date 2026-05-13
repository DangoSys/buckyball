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
class P2EBaseConfig
    extends Config(
      new WithP2EHarness ++
        new WithSCU ++
        new WithP2EDDR4MemPort ++
        new WithP2EBootROM
    )

class P2EToyConfig
    extends Config(
      new P2EBaseConfig ++
        new examples.toy.BuckyballToyConfig
    )

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
