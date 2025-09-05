package sims.firesim

import chisel3._

import org.chipsalliance.cde.config.{Config}
import freechips.rocketchip.tile._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.subsystem._

class FireSimGemminiRocketConfig extends Config(
  new firechip.chip.WithDefaultFireSimBridges ++
  new firechip.chip.WithFireSimConfigTweaks ++
  new chipyard.GemminiRocketConfig)

class FireSimBuckyballToyConfig extends Config(
  new firechip.chip.WithDefaultFireSimBridges ++
  new firechip.chip.WithFireSimConfigTweaks ++
  new examples.toy.BuckyBallToyConfig)
