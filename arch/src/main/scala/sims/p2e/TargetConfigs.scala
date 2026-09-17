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

class P2EBaseConfig(maxHarts: Int = 64)
    extends Config(
      new WithP2EHarness ++
        new WithSCU(maxHarts = maxHarts) ++
        new WithP2EDDR4MemPort ++
        new WithP2EBootROM
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

  val firtoolOpts = args.drop(1)

  val outDir = firtoolOpts.collectFirst {
    case opt if opt.startsWith("-o=") => opt.stripPrefix("-o=")
  }.getOrElse {
    throw new Exception("missing -o=<dir> in firtool opts")
  }

  ChiselStage.emitSystemVerilogFile(
    new P2EHarness()(config.toInstance),
    firtoolOpts = firtoolOpts,
    args = Array("--target-dir", outDir)
  )
  ChiselStage.emitSystemVerilogFile(
    new P2ETop,
    firtoolOpts = firtoolOpts,
    args = Array("--target-dir", outDir)
  )
}
