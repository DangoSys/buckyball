package sims.palladium

import org.chipsalliance.cde.config.Config
import chipyard._

class BuckyballToyP2EConfig extends Config(
  new palladium.fpga.WithFPGAFrequency(50) ++
  new palladium.fpga.WithVCU19PTweaks ++
  new examples.toy.BuckyBallToy1024Config
)
