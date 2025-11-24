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
  // Select Ball type from command line arguments
  val configName = if (args.isEmpty) {
    println("Usage: Elaborate <configName> [firtool-opts...]")
    println("Available config types: toy, gemmini")
    println("Using default: toy")
    "toy"
  } else {
    args(0).toLowerCase match {
      case "toy" => "toy"
      case "gemmini" => "gemmini"
      case other =>
        println(s"Unknown config name: $other, using toy")
        "toy"
    }
  }

  // Select corresponding Config based on configuration name
  val config: Config = configName match {
    case "toy" => new BuckyballToyVerilatorConfig
    case "gemmini" => new BuckyballGemminiVerilatorConfig
    // Default to toy
    case _ => new BuckyballToyVerilatorConfig
  }

  println(s"Elaborating with config: $configName")

  ChiselStage.emitSystemVerilogFile(
    new chipyard.harness.TestHarness()(config.toInstance),
    // Remaining parameters passed to firtool
    firtoolOpts = args.drop(1),
    args = Array.empty
  )
}
