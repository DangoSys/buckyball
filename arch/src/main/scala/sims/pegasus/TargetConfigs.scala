package sims.pegasus

import _root_.circt.stage.ChiselStage
import org.chipsalliance.cde.config.Config

import freechips.rocketchip.devices.tilelink.{BootROMLocated, BootROMParams}
import freechips.rocketchip.subsystem.{InSubsystem, WithDefaultMMIOPort}

class WithPegasusBootROM
    extends Config((site, here, up) => {
      case BootROMLocated(InSubsystem) => Some(BootROMParams(
          contentFileName = "src/main/resources/bootrom/bootrom.rv64.img"
        ))
    })

class PegasusConfig
    extends Config(
      new WithPegasusHarness ++
        new WithDefaultMMIOPort ++
        new WithPegasusBootROM ++
        new examples.toy.BuckyballToyConfig
    )

class PegasusBuckyballToyConfig extends PegasusConfig

object ElaboratePegasus extends App {
  ChiselStage.emitSystemVerilogFile(
    new PegasusHarness()(new PegasusConfig().toInstance),
    firtoolOpts = args,
    args = Array.empty
  )
}
