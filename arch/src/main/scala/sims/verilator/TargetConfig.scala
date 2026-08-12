package sims.verilator

import chisel3._
import _root_.circt.stage.ChiselStage
import org.chipsalliance.cde.config.Config
import freechips.rocketchip.devices.tilelink.{BootROMLocated, BootROMParams}
import freechips.rocketchip.subsystem.InSubsystem

class WithCustomBootROM
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

//===----------------------------------------------------------------------===//
// Chipyard Verilator configs
//===----------------------------------------------------------------------===//
class ChipyardRocketVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new testchipip.serdes.WithNoSerialTL ++
        new chipyard.config.WithNoUART ++
        new chipyard.config.WithNoDebug ++
        new chipyard.RocketConfig
    )

class ChipyardGemminiRocketVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new testchipip.serdes.WithNoSerialTL ++
        new chipyard.config.WithNoUART ++
        new chipyard.config.WithNoDebug ++
        new chipyard.RocketConfig
    )

class Chipyard2CoreVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new testchipip.serdes.WithNoSerialTL ++
        new chipyard.config.WithNoUART ++
        new chipyard.config.WithNoDebug ++
        new chipyard.DualRocketConfig
    )

class Chipyard4CoreVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new testchipip.serdes.WithNoSerialTL ++
        new chipyard.config.WithNoUART ++
        new chipyard.config.WithNoDebug ++
        new chipyard.QuadRocketConfig
    )

class Chipyard8CoreVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new testchipip.serdes.WithNoSerialTL ++
        new chipyard.config.WithNoUART ++
        new chipyard.config.WithNoDebug ++
        new freechips.rocketchip.rocket.WithNHugeCores(8) ++
        new chipyard.config.AbstractConfig
    )

class Chipyard4CoreGemminiVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new testchipip.serdes.WithNoSerialTL ++
        new chipyard.config.WithNoUART ++
        new chipyard.config.WithNoDebug ++
        new freechips.rocketchip.rocket.WithNHugeCores(4) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )

class Chipyard8CoreGemminiVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new testchipip.serdes.WithNoSerialTL ++
        new chipyard.config.WithNoUART ++
        new chipyard.config.WithNoDebug ++
        new freechips.rocketchip.rocket.WithNHugeCores(8) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )

class Chipyard32CoreGemminiVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new testchipip.serdes.WithNoSerialTL ++
        new chipyard.config.WithNoUART ++
        new chipyard.config.WithNoDebug ++
        new freechips.rocketchip.rocket.WithNHugeCores(32) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )

class Chipyard64CoreGemminiVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new testchipip.serdes.WithNoSerialTL ++
        new chipyard.config.WithNoUART ++
        new chipyard.config.WithNoDebug ++
        new freechips.rocketchip.rocket.WithNHugeCores(64) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )

//===----------------------------------------------------------------------===//
object Elaborate extends App {
  if (args.isEmpty) {
    println("Usage: Elaborate <full.config.ClassName> [firtool-opts...]")
    println("Example: Elaborate sims.verilator.BuckyballToyVerilatorConfig")
    sys.exit(1)
  }
  val configClassName = args(0)
  println(s"Elaborating BBSimHarness with config: $configClassName")

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
    new BBSimHarness()(config.toInstance),
    firtoolOpts = args.drop(1),
    args = Array.empty
  )
}
