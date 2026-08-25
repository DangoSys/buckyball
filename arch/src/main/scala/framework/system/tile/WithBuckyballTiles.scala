package framework.system.tile

import org.chipsalliance.cde.config.{Config, Parameters}
import freechips.rocketchip.subsystem.{CoherenceManagerWrapper, SubsystemBankedCoherenceKey}
import framework.system.configloader.ChipLoader

/**
 * Build an N-BBTile chipyard subsystem from chip.pb.
 */
class WithBuckyballTiles(
  pbPath:         String,
  withBuckyball:  Boolean = true,
  hiddenHartBase: Option[Int] = None)
    extends Config(WithBuckyballTiles.assemble(pbPath, withBuckyball, hiddenHartBase))

object WithBuckyballTiles {

  def assemble(pbPath: String, withBuckyball: Boolean, hiddenHartBase: Option[Int]): Parameters = {
    if (!pbPath.endsWith(".pb")) {
      throw new RuntimeException(s"WithBuckyballTiles expects a chip.pb path, got: $pbPath")
    }
    val topology = ChipLoader.load(pbPath)

    val tileFragments: Seq[Config] = topology.tiles.map { tile =>
      val resolved = if (withBuckyball) tile.cores else tile.cores.map(_ => None)
      new WithBBTile(
        withBuckyball = resolved.exists(_.isDefined),
        nCoresPerTile = tile.cores.size,
        buckyballPerCore = Some(resolved),
        rocketCorePerCore = Some(tile.rocketCores),
        privateDCache = tile.privateDCache,
        hiddenHartBase = hiddenHartBase
      )
    }

    val anyPrivateDCache = topology.tiles.exists(_.privateDCache.isDefined)
    val coherenceFragment: Seq[Config] =
      if (anyPrivateDCache) Seq(new WithIncoherentSystemBus) else Nil

    (tileFragments ++ coherenceFragment).reduce[Parameters](_ ++ _)
  }

}

class WithIncoherentSystemBus
    extends Config((site, here, up) => {
      case SubsystemBankedCoherenceKey => up(SubsystemBankedCoherenceKey, site).copy(
          coherenceManager = CoherenceManagerWrapper.incoherentManager
        )
    })
