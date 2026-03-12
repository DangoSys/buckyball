package sims.verilator

import chisel3._
import _root_.circt.stage.ChiselStage
import org.chipsalliance.cde.config.{Config, Parameters}

import freechips.rocketchip.devices.tilelink.{BootROMLocated, BootROMParams}
import freechips.rocketchip.subsystem.InSubsystem

class WithCustomBootROM
    extends Config((site, here, up) => {
      case BootROMLocated(InSubsystem) => Some(BootROMParams(
          contentFileName = "src/main/resources/bootrom/bootrom.rv64.img"
        ))
    })

class BuckyballToyVerilatorConfig
    extends Config(
      new WithCustomBootROM ++
        new examples.toy.BuckyballToyConfig
    )

class BuckyballGobanVerilatorConfig
    extends Config(
      new WithCustomBootROM ++
        new examples.goban.BuckyballGobanConfig
    )

class BuckyballGemminiVerilatorConfig
    extends Config(
      new WithCustomBootROM ++
        new gemmini.DefaultGemminiConfig
    )

object Elaborate extends App {
  if (args.isEmpty) {
    println("Usage: Elaborate <full.config.ClassName> [firtool-opts...]")
    println("Example: Elaborate sims.verilator.BuckyballToyVerilatorConfig")
    sys.exit(1)
  }

  val configClassName = args(0)
  println(s"Elaborating with config class: $configClassName")

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
    new chipyard.harness.TestHarness()(config.toInstance),
    // Remaining parameters passed to firtool
    firtoolOpts = args.drop(1),
    args = Array.empty
  )
}

// BBSimElaborate: elaborates BBSimHarness instead of TestHarness.
// Used for bdb-based simulation (no fesvr, ELF loaded via libelf).
// Usage: BBSimElaborate sims.verilator.BuckyballToyBBSimConfig [firtool-opts...]
object BBSimElaborate extends App {
  if (args.isEmpty) {
    println("Usage: BBSimElaborate <full.config.ClassName> [firtool-opts...]")
    println("Example: BBSimElaborate sims.verilator.BuckyballToyBBSimConfig")
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
