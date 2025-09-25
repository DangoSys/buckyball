package sims.firesim

import chisel3._
import java.io.File

import org.chipsalliance.cde.config.{Config}
import freechips.rocketchip.tile._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.subsystem._
import freechips.rocketchip.devices.tilelink.{BootROMParams, BootROMLocated}

class WithBootROM extends Config((site, here, up) => {
  case BootROMLocated(x) => {
    val chipyardBootROM = new File(s"./thirdparty/chipyard/generators/testchipip/bootrom/bootrom.rv${site(MaxXLen)}.img")
    val firesimBootROM = new File(s"./thirdparty/chipyard/target-rtl/chipyard/generators/testchipip/bootrom/bootrom.rv${site(MaxXLen)}.img")

    val bootROMPath = if (chipyardBootROM.exists()) {
      chipyardBootROM.getAbsolutePath()
    } else {
      firesimBootROM.getAbsolutePath()
    }
    up(BootROMLocated(x)).map(_.copy(contentFileName = bootROMPath))
  }
})

class FireSimGemminiBuckyballConfig extends Config(
  new WithBootROM ++
  new firechip.chip.WithDefaultFireSimBridges ++
  new firechip.chip.WithFireSimConfigTweaks ++
  new chipyard.GemminiRocketConfig)

class FireSimBuckyballToyConfig extends Config(
  new WithBootROM ++
  new firechip.chip.WithDefaultFireSimBridges ++
  new firechip.chip.WithFireSimConfigTweaks ++
  new examples.toy.BuckyBallToyConfig)
