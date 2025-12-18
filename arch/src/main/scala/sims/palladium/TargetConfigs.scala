package sims.palladium

import org.chipsalliance.cde.config.Config
import chipyard._

class BuckyballToyP2EConfig extends Config(
  new palladium.fpga.WithFPGAFrequency(50) ++
  new palladium.fpga.WithVCU19PTweaks ++
  new examples.toy.BuckyballToy256Config  // Test with 256 cores + 8 L2 banks
)

// Cross-bar config
class BuckyballToyP2ECBConfig extends Config(
  new palladium.fpga.WithFPGAFrequency(50) ++
  new palladium.fpga.WithVCU19PTweaks ++
  new examples.toy.BuckyballToy256CBConfig
)
