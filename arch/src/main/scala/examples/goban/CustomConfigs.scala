package examples.goban

import org.chipsalliance.cde.config.Config
import framework.core.bbtile.WithNBBTiles
import framework.top.GlobalConfig
import framework.top.configs.TopConfig

import freechips.rocketchip.subsystem.InSubsystem

/**
 * Goban: multi-core BBTile configuration.
 *
 * Each BBTile contains nCores Rocket cores, each paired with a BuckyballAccelerator.
 * All accelerators within the tile share a single SharedMemBackend and BarrierUnit.
 * The ISA, Ball operators, and memory layout are identical to the toy configuration.
 */
object GobanConfig {
  /** Number of cores inside each BBTile. */
  val nCores: Int = 4

  /** GlobalConfig for goban: same as toy but with nCores set. */
  def apply(): GlobalConfig = {
    val base = GlobalConfig()
    base.copy(top = base.top.copy(nCores = nCores))
  }
}

/** 1 BBTile × 4 cores (4 Rocket + 4 Buckyball, shared SharedMem + BarrierUnit) */
class BuckyballGobanConfig extends Config(
  new WithNBBTiles(1, buckyballConfig = GobanConfig()) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig
)

/** 2 BBTiles × 4 cores = 8 total cores */
class BuckyballGoban2TileConfig extends Config(
  new WithNBBTiles(2, buckyballConfig = GobanConfig()) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig
)
