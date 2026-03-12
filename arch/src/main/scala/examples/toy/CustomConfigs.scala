package examples.toy

import org.chipsalliance.cde.config.Config
import framework.core.bbtile.WithNBBTiles

import freechips.rocketchip.subsystem.{InCluster, InSubsystem}
import freechips.rocketchip.devices.tilelink.{BootROMLocated, BootROMParams}
import constellation.channel._
import constellation.routing._
import constellation.router._
import constellation.topology._
import constellation.noc._
import scala.collection.immutable.ListMap

/** Single BBTile: 1 Rocket core + 1 Buckyball accelerator */
class BuckyballToyConfig
    extends Config(
      new WithNBBTiles(1) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )

/** Single Rocket core only (no Buckyball) */
class RocketOnlyConfig
    extends Config(
      new WithNBBTiles(1, withBuckyball = false) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )

// Increase BootROM size for large core counts
class WithLargeBootROM(address: BigInt = 0x80000, size: Int = 0x80000)
    extends Config((site, here, up) => {
      case BootROMLocated(InSubsystem) =>
        up(BootROMLocated(InSubsystem)).map(_.copy(address = address, size = size))
    })
