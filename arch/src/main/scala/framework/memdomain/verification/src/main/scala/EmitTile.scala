package framework.memdomain.verification

import java.nio.file.{Files, Path}
import chisel3._
import chisel3.experimental.hierarchy.Instance
import framework.system.{System, SystemParams}
import framework.system.configloader.ExampleTopology
import memcore.bus.chi.{Params => ChiParams}
import memcore.bus.chi.rnf.RnfParams
import memcore.memory.cache.configs.CacheParams
import memcore.memory.coherence.configs.CoherenceParams
import memcore.memory.cpu.PhysicalRegion
import memcore.memory.interlock.{Params => TrackingParams}

/** Elaboration of the actual single-tile composition, with explicit verification cache geometry. */
object EmitTile {

  def apply(
    topology:       ExampleTopology,
    output:         Path,
    firtoolOptions: Array[String],
    build:          SystemParams => Instance[System]
  ): Unit = {
    require(topology.tiles.size == 1, "Tile gate requires one configured tile")
    val t              = topology.tiles.head
    val chi            = ChiParams()
    val memory         = CoherenceParams(
      chi,
      CacheParams(chi.addressBits, 64, 4, 2, 8, 4, 2),
      agents = 2 * t.cores.size,
      mshrEntries = 8,
      homeId = 64
    )
    val regions        = Seq(
      PhysicalRegion(BigInt("80000000", 16), BigInt(256) << 20, true, true, true, true, true, true),
      PhysicalRegion(BigInt("90000000", 16), BigInt(16) << 20, false, true, true, true, true, true),
      PhysicalRegion(BigInt("10000000", 16), BigInt(4096), false, false, true, true, false, false),
      PhysicalRegion(BigInt("60000000", 16), BigInt(4096), false, false, true, true, false, false)
    )
    val slotsPerClient = 16
    val ddr            = memcore.memory.ddr.Params(
      chi,
      clients = 1,
      slotsPerClient = slotsPerClient,
      dataBits = 128,
      idBits = 4
    )
    var interface: Data = null
    _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
      {
        val system = new TileSystem(
          topology,
          36,
          memory,
          RnfParams(chi, cacheLines = 8, banks = 2),
          regions,
          TrackingParams(addressBits = chi.addressBits),
          ddr,
          build
        )
        interface = system.io
        system
      },
      args = Array("--target-dir", output.resolve("TileSystem").toString, "--split-verilog"),
      firtoolOpts = firtoolOptions
    )
    def leaves(data: Data, name: String): Seq[(String, Data)] = data match {
      case record: Record => record.elements.toSeq.flatMap { case (n, v) => leaves(v, s"${name}_$n") }
      case vector: Vec[_] => vector.zipWithIndex.flatMap { case (value, i) => leaves(value, s"${name}_$i") }.toSeq
      case leaf => if (leaf.getWidth > 0) Seq(name -> leaf) else Seq.empty
    }
    val fields = leaves(interface, "io")
    Files.writeString(
      output.resolve("tile_system_signals.svh"),
      fields.map {
        case (name, data) => s"logic [${data.getWidth - 1}:0] $name;"
      }.mkString("\n") + "\n"
    )
    Files.writeString(
      output.resolve("tile_system_ports.svh"),
      (Seq(".clock(clock)", ".reset(reset)") ++ fields.map { case (name, _) => s".$name($name)" }).mkString(
        ",\n"
      ) + "\n"
    )
    Files.writeString(output.resolve("tile_system_config.svh"), s"`define TILE_CPU_COUNT ${t.cores.size}\n")
  }

}
