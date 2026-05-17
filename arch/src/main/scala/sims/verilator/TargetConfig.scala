package sims.verilator

import chisel3._
import _root_.circt.stage.ChiselStage
import org.chipsalliance.cde.config.Config

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
      new BBSimConfig ++
        new WithCustomBootROM ++
        new examples.toy.BuckyballToyConfig
    )

//===----------------------------------------------------------------------===//
// Goban Verilator configs
//===----------------------------------------------------------------------===//
class BuckyballGoban2CoreVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new examples.goban.BuckyballGoban2CoreConfig
    )

class BuckyballGoban4CoreVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new examples.goban.BuckyballGoban4CoreConfig
    )

class BuckyballGoban8CoreVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new examples.goban.BuckyballGoban8CoreConfig
    )

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Chiplet Verilator configs
//===----------------------------------------------------------------------===//
class BuckyballChiplet2CoreVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new examples.chiplet.BuckyballChiplet2CoreConfig
    )

class BuckyballChiplet4CoreVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new examples.chiplet.BuckyballChiplet4CoreConfig
    )

class BuckyballChiplet8CoreVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new examples.chiplet.BuckyballChiplet8CoreConfig
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
