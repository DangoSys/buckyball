package examples.toy

import org.chipsalliance.cde.config.{Config, Field, Parameters}
import chisel3._
import freechips.rocketchip.diplomacy.LazyModule
import freechips.rocketchip.subsystem.SystemBusKey
import freechips.rocketchip.tile._
import examples.toy.ToyBuckyball
import framework.builtin.BaseConfig
import examples.BuckyballConfigs.CustomBuckyballConfig
import examples.CustomBuckyballConfig

object BuckyballToyConfig {

  val defaultConfig = new BaseConfig(
    bankNum = 32
  )

}

class BuckyballCustomConfig(
  buckyballConfig: CustomBuckyballConfig = CustomBuckyballConfig())
    extends Config((site, here, up) => {
      case BuildRoCC => up(BuildRoCC) ++ Seq {
          (p: Parameters) =>
            implicit val q = p
            val buckyball  = LazyModule(new ToyBuckyball(buckyballConfig)(q))
            buckyball
        }
    })

class BuckyballToyConfig
    extends Config(
      new BuckyballCustomConfig ++
        new framework.core.rocket.WithNBuckyballCores(1) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )

import freechips.rocketchip.subsystem.{InCluster, InSubsystem, MBUS, SBUS}
import freechips.rocketchip.devices.tilelink.{BootROMLocated, BootROMParams}
import constellation.channel._
import constellation.routing._
import constellation.router._
import constellation.topology._
import constellation.noc._
import scala.collection.immutable.ListMap

// Increase BootROM size for large core counts (device tree becomes very large)
// Note: For AddressSet(base, size-1), we need (base & (size-1)) == 0
// This means base must be aligned to size (size must be power of 2)
// For 256 cores (8 clusters × 32 cores), 512KB should be sufficient
class WithLargeBootROM(address: BigInt = 0x80000, size: Int = 0x80000)
    extends Config((site, here, up) => {
      case BootROMLocated(InSubsystem) => {
        up(BootROMLocated(InSubsystem)).map(_.copy(address = address, size = size))
      }
    })

// 4-core test configuration to understand NoC mapping
class BuckyballToy4Config
    extends Config(
      new WithLargeBootROM(0x80000, 0x80000) ++
        new constellation.soc.WithSbusNoC(constellation.protocol.SimpleTLNoCParams(
          constellation.protocol.DiplomaticNetworkNodeMapping(
            inNodeMapping = ListMap(
              "serial_tl"                                             -> 0,
              "Core 0 "                                               -> 1, // Space after number for precise matching
              "Core 1 "                                               -> 2,
              "Core 2 "                                               -> 3,
              "Core 3 "                                               -> 4,
              "debug"                                                 -> 5,
              // buckyball-stream ports appear together, map them to same node
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
            topology = TerminalRouter(Mesh2D(4, 4)), // 4x4 mesh = 16 nodes (enough for 10 inputs + 5 outputs)
            channelParamGen = (a, b) => UserChannelParams(Seq.fill(8)(UserVirtualChannelParams(4))),
            routingRelation = BlockingVirtualSubnetworksRouting(TerminalRouterRouting(Mesh2DEscapeRouting()), 5, 1)
          )
        )) ++
        new BuckyballCustomConfig ++
        new framework.core.rocket.WithNBuckyballCores(4) ++
        new freechips.rocketchip.subsystem.WithNBanks(4) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )

// 64-core test configuration with 2 L2 banks
class BuckyballToy64Config
    extends Config(
      new WithLargeBootROM(0x80000, 0x80000) ++
        new constellation.soc.WithSbusNoC(constellation.protocol.SimpleTLNoCParams(
          constellation.protocol.DiplomaticNetworkNodeMapping(
            inNodeMapping = ListMap(
              "serial_tl" -> 0,
              "debug"     -> 1
            ) ++ (0 until 64).map(i => s"Core $i " -> (2 + i)).toMap // Note the space after number!
              ++ (0 until 64).map(i => s"buckyball-stream-reader[$i],buckyball-stream-writer[$i]" -> (66 + i)).toMap,
            outNodeMapping = ListMap(
              "pbus" -> 130
            ) ++ (0 until 2).map(i => s"system[$i]" -> (131 + i)).toMap
          ),
          NoCParams(
            // 12x12 mesh = 144 nodes (enough for 130 inputs + 3 outputs)
            topology = TerminalRouter(Mesh2D(12, 12)),
            channelParamGen = (a, b) => UserChannelParams(Seq.fill(8)(UserVirtualChannelParams(4))),
            routingRelation = BlockingVirtualSubnetworksRouting(TerminalRouterRouting(Mesh2DEscapeRouting()), 5, 1)
          )
        )) ++
        new BuckyballCustomConfig ++
        new framework.core.rocket.WithNBuckyballCores(64) ++
        new freechips.rocketchip.subsystem.WithNBanks(2) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )

// 256-core test configuration with 32 L2 banks
class BuckyballToy256Config
    extends Config(
      new WithLargeBootROM(0x80000, 0x80000) ++
        new constellation.soc.WithSbusNoC(constellation.protocol.SimpleTLNoCParams(
          constellation.protocol.DiplomaticNetworkNodeMapping(
            inNodeMapping = ListMap(
              "serial_tl" -> 0,
              "debug"     -> 1
            ) ++ (0 until 256).map(i => s"Core $i " -> (2 + i)).toMap // Note the space after number!
              ++ (0 until 256).map(i => s"buckyball-stream-reader[$i],buckyball-stream-writer[$i]" -> (258 + i)).toMap,
            outNodeMapping = ListMap(
              "pbus" -> 514
            ) ++ (0 until 32).map(i => s"system[$i]" -> (515 + i)).toMap
          ),
          NoCParams(
            // 24x24 mesh = 576 nodes (enough for 514 inputs + 33 outputs)
            topology = TerminalRouter(Mesh2D(24, 24)),
            channelParamGen = (a, b) => UserChannelParams(Seq.fill(8)(UserVirtualChannelParams(4))),
            routingRelation = BlockingVirtualSubnetworksRouting(TerminalRouterRouting(Mesh2DEscapeRouting()), 5, 1)
          )
        )) ++
        new BuckyballCustomConfig ++
        new framework.core.rocket.WithNBuckyballCores(256) ++
        new freechips.rocketchip.subsystem.WithNBanks(32) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )

class BuckyballToy256CBConfig
    extends Config(
      new WithLargeBootROM(0x80000, 0x80000) ++ // 512KB BootROM at 0x80000 (for 1024 cores)
        new BuckyballCustomConfig ++
        new framework.core.rocket.WithNBuckyballCores(32, location = InCluster(7)) ++
        new framework.core.rocket.WithNBuckyballCores(32, location = InCluster(6)) ++
        new framework.core.rocket.WithNBuckyballCores(32, location = InCluster(5)) ++
        new framework.core.rocket.WithNBuckyballCores(32, location = InCluster(4)) ++
        new framework.core.rocket.WithNBuckyballCores(32, location = InCluster(3)) ++
        new framework.core.rocket.WithNBuckyballCores(32, location = InCluster(2)) ++
        new framework.core.rocket.WithNBuckyballCores(32, location = InCluster(1)) ++
        new framework.core.rocket.WithNBuckyballCores(32, location = InCluster(0)) ++
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
