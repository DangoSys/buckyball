package sims.verilator

import chisel3._
// _root_ disambiguates from package chisel3.util.circt if user imports chisel3.util._
import _root_.circt.stage.ChiselStage
import org.chipsalliance.cde.config.{Config, Parameters}

import freechips.rocketchip.devices.tilelink.{BootROMParams, BootROMLocated}
import freechips.rocketchip.subsystem.InSubsystem


// Custom BootROM configuration, pointing to correct resource path
class WithCustomBootROM extends Config((site, here, up) => {
  case BootROMLocated(InSubsystem) => Some(BootROMParams(
    contentFileName = "src/main/resources/bootrom/bootrom.rv64.img"
  ))
})

class BuckyballToyVerilatorConfig extends Config(
  new WithCustomBootROM ++
  new examples.toy.BuckyballToyConfig)

class BuckyballGemminiVerilatorConfig extends Config(
  new WithCustomBootROM ++
  new gemmini.DefaultGemminiConfig)



object Elaborate extends App {
  // Accept full config class name like "sims.verilator.BuckyballToyVerilatorConfig"
  if (args.isEmpty) {
    println("Usage: Elaborate <full.config.ClassName> [firtool-opts...]")
    println("Example: Elaborate sims.verilator.BuckyballToyVerilatorConfig")
    sys.exit(1)
  }

  val configClassName = args(0)
  println(s"Elaborating with config class: $configClassName")

  // Dynamically load the config class
  val config: Config = try {
    val configClass = Class.forName(configClassName)
    configClass.getDeclaredConstructor().newInstance().asInstanceOf[Config]
  } catch {
    case e: ClassNotFoundException =>
      println(s"Error: Config class not found: $configClassName")
      sys.exit(1)
    case e: Exception =>
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
