package sims.p2e

import chisel3._
import _root_.circt.stage.ChiselStage
import org.chipsalliance.cde.config.Config

import freechips.rocketchip.devices.tilelink.{BootROMLocated, BootROMParams}
import freechips.rocketchip.subsystem.{InSubsystem, WithCustomMMIOPort, WithCustomMemPort}
import sims.p2e.scu.WithP2ESCU

class WithP2EBootROM
    extends Config((site, here, up) => {
      case BootROMLocated(InSubsystem) => Some(BootROMParams(
          contentFileName = "src/main/resources/bootrom/bootrom.rv64.img"
        ))
    })

class WithP2EDDR4MemPort
    extends Config(
      new WithCustomMemPort(
        base_addr = BigInt("80000000", 16),
        base_size = BigInt("10000000", 16),
        data_width = 256,
        id_bits = 11,
        maxXferBytes = 256
      )
    )

class P2EBaseConfig
    extends Config(
      new WithP2EHarness ++
        new WithP2ESCU ++
        new WithCustomMMIOPort(
          base_addr = BigInt("60040000", 16),
          base_size = BigInt("1ffc0000", 16),
          data_width = 64,
          id_bits = 4,
          maxXferBytes = 64
        ) ++
        new WithP2EDDR4MemPort ++
        new WithP2EBootROM
    )

class P2EToyConfig
    extends Config(
      new P2EBaseConfig ++
        new examples.toy.BuckyballToyConfig
    )

object Elaborate extends App {
  if (args.isEmpty) {
    println("Usage: Elaborate <full.config.ClassName> [firtool-opts...]")
    println("Example: Elaborate sims.p2e.P2EToyConfig")
    sys.exit(1)
  }

  val configClassName = args(0)
  println(s"Elaborating P2EHarness with config: $configClassName")

  val config: Config =
    try {
      val configClass = Class.forName(configClassName)
      configClass.getDeclaredConstructor().newInstance().asInstanceOf[Config]
    } catch {
      case e: ClassNotFoundException =>
        println(s"Error: Config class not found: $configClassName")
        sys.exit(1)
      case e: Exception              =>
        println(s"Error loading config class: ${e.getMessage}")
        e.printStackTrace()
        sys.exit(1)
    }

  ChiselStage.emitSystemVerilogFile(
    new P2EHarness()(config.toInstance),
    firtoolOpts = args.drop(1),
    args = Array.empty
  )
}
