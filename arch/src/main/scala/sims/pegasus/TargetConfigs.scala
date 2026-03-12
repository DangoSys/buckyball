package sims.pegasus

import chisel3._
import _root_.circt.stage.ChiselStage
import org.chipsalliance.cde.config.Config

import freechips.rocketchip.devices.tilelink.{BootROMLocated, BootROMParams}
import freechips.rocketchip.subsystem.InSubsystem

// import pegasus.{PegasusHarness, WithPegasusHarness}

// BootROM for FPGA (points to the same bootrom image as the Verilator target)
class WithPegasusBootROM
    extends Config((site, here, up) => {
      case BootROMLocated(InSubsystem) => Some(BootROMParams(
        contentFileName = "src/main/resources/bootrom/bootrom.rv64.img"
      ))
    })

// PegasusBuckyballToyConfig: Buckyball Toy SoC on AU280 FPGA
//
// Target clock: 200 MHz. If timing closure yields a different Fmax,
// update all WithXxxBusFrequency values and re-elaborate so the DTB
// gets the correct UART baud divisor (otherwise UART output will be garbled).
class PegasusBuckyballToyConfig extends Config(
  new WithPegasusHarness ++
  new WithPegasusBootROM ++
  new chipyard.config.WithSystemBusFrequency(200.0) ++
  new chipyard.config.WithMemoryBusFrequency(200.0) ++
  new chipyard.config.WithPeripheryBusFrequency(200.0) ++
  new chipyard.config.WithControlBusFrequency(200.0) ++
  new chipyard.config.WithFrontBusFrequency(200.0) ++
  new chipyard.config.WithOffchipBusFrequency(200.0) ++
  new examples.toy.BuckyballToyConfig
)

// Elaborate entry point: generate SystemVerilog for PegasusHarness
// Usage: mill pegasus.runMain sims.pegasus.ElaboratePegasus [firtool options]
object ElaboratePegasus extends App {
  ChiselStage.emitSystemVerilogFile(
    new PegasusHarness()(new PegasusBuckyballToyConfig().toInstance),
    firtoolOpts = args,
    args = Array.empty
  )
}
