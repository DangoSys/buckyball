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

class BuckyballToy4Config
    extends Config(
      new WithLargeBootROM(0x80000, 0x80000) ++
        new constellation.soc.WithSbusNoC(constellation.protocol.SimpleTLNoCParams(
          constellation.protocol.DiplomaticNetworkNodeMapping(
            inNodeMapping = ListMap(
              "serial_tl"                                             -> 0,
              "Core 0 "                                               -> 1,
              "Core 1 "                                               -> 2,
              "Core 2 "                                               -> 3,
              "Core 3 "                                               -> 4,
              "debug"                                                 -> 5,
              "buckyball-stream-reader[0],buckyball-stream-writer[0]" -> 6,
              "buckyball-stream-reader[1],buckyball-stream-writer[1]" -> 7,
              "buckyball-stream-reader[2],buckyball-stream-writer[2]" -> 8,
              "buckyball-stream-reader[3],buckyball-stream-writer[3]" -> 9
            ),
            outNodeMapping = ListMap(
              "pbus"      -> 10,
              "system[0]" -> 11,
              "system[1]" -> 12,
              "system[2]" -> 13,
              "system[3]" -> 14
            )
          ),
          NoCParams(
            topology = TerminalRouter(Mesh2D(4, 4)),
            channelParamGen = (a, b) => UserChannelParams(Seq.fill(8)(UserVirtualChannelParams(4))),
            routingRelation = BlockingVirtualSubnetworksRouting(TerminalRouterRouting(Mesh2DEscapeRouting()), 5, 1)
          )
        )) ++
        new WithNBBTiles(4) ++
        new freechips.rocketchip.subsystem.WithNBanks(4) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )

class BuckyballToy64Config
    extends Config(
      new WithLargeBootROM(0x80000, 0x80000) ++
        new constellation.soc.WithSbusNoC(constellation.protocol.SimpleTLNoCParams(
          constellation.protocol.DiplomaticNetworkNodeMapping(
            inNodeMapping = ListMap(
              "serial_tl" -> 0,
              "debug"     -> 1
            ) ++ (0 until 64).map(i => s"Core $i " -> (2 + i)).toMap
              ++ (0 until 64).map(i => s"buckyball-stream-reader[$i],buckyball-stream-writer[$i]" -> (66 + i)).toMap,
            outNodeMapping = ListMap(
              "pbus" -> 130
            ) ++ (0 until 2).map(i => s"system[$i]" -> (131 + i)).toMap
          ),
          NoCParams(
            topology = TerminalRouter(Mesh2D(12, 12)),
            channelParamGen = (a, b) => UserChannelParams(Seq.fill(8)(UserVirtualChannelParams(4))),
            routingRelation = BlockingVirtualSubnetworksRouting(TerminalRouterRouting(Mesh2DEscapeRouting()), 5, 1)
          )
        )) ++
        new WithNBBTiles(64) ++
        new freechips.rocketchip.subsystem.WithNBanks(2) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )

class BuckyballToy256Config
    extends Config(
      new WithLargeBootROM(0x80000, 0x80000) ++
        new constellation.soc.WithSbusNoC(constellation.protocol.SimpleTLNoCParams(
          constellation.protocol.DiplomaticNetworkNodeMapping(
            inNodeMapping = ListMap(
              "serial_tl" -> 0,
              "debug"     -> 1
            ) ++ (0 until 256).map(i => s"Core $i " -> (2 + i)).toMap
              ++ (0 until 256).map(i => s"buckyball-stream-reader[$i],buckyball-stream-writer[$i]" -> (258 + i)).toMap,
            outNodeMapping = ListMap(
              "pbus" -> 514
            ) ++ (0 until 32).map(i => s"system[$i]" -> (515 + i)).toMap
          ),
          NoCParams(
            topology = TerminalRouter(Mesh2D(24, 24)),
            channelParamGen = (a, b) => UserChannelParams(Seq.fill(8)(UserVirtualChannelParams(4))),
            routingRelation = BlockingVirtualSubnetworksRouting(TerminalRouterRouting(Mesh2DEscapeRouting()), 5, 1)
          )
        )) ++
        new WithNBBTiles(256) ++
        new freechips.rocketchip.subsystem.WithNBanks(32) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )

class BuckyballToy256CBConfig
    extends Config(
      new WithLargeBootROM(0x80000, 0x80000) ++
        new WithNBBTiles(32, location = InCluster(7)) ++
        new WithNBBTiles(32, location = InCluster(6)) ++
        new WithNBBTiles(32, location = InCluster(5)) ++
        new WithNBBTiles(32, location = InCluster(4)) ++
        new WithNBBTiles(32, location = InCluster(3)) ++
        new WithNBBTiles(32, location = InCluster(2)) ++
        new WithNBBTiles(32, location = InCluster(1)) ++
        new WithNBBTiles(32, location = InCluster(0)) ++
        new freechips.rocketchip.subsystem.WithCluster(7) ++
        new freechips.rocketchip.subsystem.WithCluster(6) ++
        new freechips.rocketchip.subsystem.WithCluster(5) ++
        new freechips.rocketchip.subsystem.WithCluster(4) ++
        new freechips.rocketchip.subsystem.WithCluster(3) ++
        new freechips.rocketchip.subsystem.WithCluster(2) ++
        new freechips.rocketchip.subsystem.WithCluster(1) ++
        new freechips.rocketchip.subsystem.WithCluster(0) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )
