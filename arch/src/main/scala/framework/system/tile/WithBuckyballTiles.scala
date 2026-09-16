package framework.system.tile

import org.chipsalliance.cde.config.{Config, Parameters}
import freechips.rocketchip.subsystem.{CoherenceManagerWrapper, SubsystemBankedCoherenceKey}
import framework.system.configloader.{BoomTileCore, ChipLoader, RocketTileCore, TileTopology}

/**
 * Build a chipyard subsystem from chip.pb.
 *
 * CPU kind is per-core. All-rocket tiles keep BBTile. Boom cores attach as BoomTiles.
 * Heterogeneous tiles attach cores in order (each rocket as BBTile n=1, each boom as BoomTile);
 * privateDCache / sharedMem / buckyball are forbidden on hetero tiles.
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

    val tileFragments: Seq[Config] = topology.tiles.flatMap { tile =>
      fragmentsForTile(tile, withBuckyball, hiddenHartBase)
    }

    val anyPrivateDCache = topology.tiles.exists(_.privateDCache.isDefined)
    val coherenceFragment: Seq[Config] =
      if (anyPrivateDCache) Seq(new WithIncoherentSystemBus) else Nil

    (tileFragments ++ coherenceFragment).reduce[Parameters](_ ++ _)
  }

  private def fragmentsForTile(
    tile:           TileTopology,
    withBuckyball:  Boolean,
    hiddenHartBase: Option[Int]
  ): Seq[Config] = {
    val allRocket = tile.cores.forall(_.isInstanceOf[RocketTileCore])
    val allBoom   = tile.cores.forall(_.isInstanceOf[BoomTileCore])
    if (allRocket) {
      Seq(rocketTile(tile, withBuckyball, hiddenHartBase))
    } else if (allBoom) {
      if (tile.privateDCache.isDefined) {
        throw new RuntimeException("boom-only tile cannot enable privateDCache")
      }
      tile.cores.map {
        case BoomTileCore(boom) => new WithBoomTile(boom)
        case other              => throw new RuntimeException(s"expected BoomTileCore, got $other")
      }
    } else {
      if (tile.privateDCache.isDefined) {
        throw new RuntimeException("heterogeneous tile cannot enable privateDCache")
      }
      tile.cores.map {
        case RocketTileCore(rocket, buckyball) =>
          if (buckyball.isDefined) {
            throw new RuntimeException("heterogeneous tile cannot enable buckyball")
          }
          new WithBBTile(
            withBuckyball = false,
            nCoresPerTile = 1,
            buckyballPerCore = Some(Seq(None)),
            rocketCorePerCore = Some(Seq(rocket)),
            privateDCache = None,
            hiddenHartBase = hiddenHartBase
          )
        case BoomTileCore(boom)                =>
          new WithBoomTile(boom)
      }
    }
  }

  private def rocketTile(
    tile:           TileTopology,
    withBuckyball:  Boolean,
    hiddenHartBase: Option[Int]
  ): Config = {
    val rockets  = tile.cores.map {
      case RocketTileCore(rocket, bb) => (rocket, bb)
      case other                      => throw new RuntimeException(s"expected RocketTileCore, got $other")
    }
    val resolved =
      if (withBuckyball) rockets.map(_._2)
      else rockets.map(_ => None)
    new WithBBTile(
      withBuckyball = resolved.exists(_.isDefined),
      nCoresPerTile = rockets.size,
      buckyballPerCore = Some(resolved),
      rocketCorePerCore = Some(rockets.map(_._1)),
      privateDCache = tile.privateDCache,
      hiddenHartBase = hiddenHartBase
    )
  }

}

class WithIncoherentSystemBus
    extends Config((site, here, up) => {
      case SubsystemBankedCoherenceKey => up(SubsystemBankedCoherenceKey, site).copy(
          coherenceManager = CoherenceManagerWrapper.incoherentManager
        )
    })
