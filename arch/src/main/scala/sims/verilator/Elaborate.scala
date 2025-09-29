package sims.verilator

import chisel3._
// _root_ disambiguates from package chisel3.util.circt if user imports chisel3.util._
import _root_.circt.stage.ChiselStage
import org.chipsalliance.cde.config.{Config, Parameters}

import freechips.rocketchip.devices.tilelink.{BootROMParams, BootROMLocated}
import freechips.rocketchip.subsystem.InSubsystem


// 自定义BootROM配置，指向正确的资源路径
class WithCustomBootROM extends Config((site, here, up) => {
  case BootROMLocated(InSubsystem) => Some(BootROMParams(
    contentFileName = "src/main/resources/bootrom/bootrom.rv64.img"
  ))
})

class BuckyBallToyVerilatorConfig extends Config(
  new WithCustomBootROM ++
  new examples.toy.BuckyBallToyConfig)

object Elaborate extends App {
  val config = new BuckyBallToyVerilatorConfig
  val params = config.toInstance

  ChiselStage.emitSystemVerilogFile(
    new chipyard.harness.TestHarness()(config.toInstance),
    firtoolOpts = args,
    args = Array.empty  // directly pass command line arguments to firtool
  )
}
