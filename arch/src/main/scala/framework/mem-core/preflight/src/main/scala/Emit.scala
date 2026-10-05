package memcore.memory.preflight

import chisel3._
import java.nio.file.{Files, Paths}
import memcore.bus.chi.rnf.{CacheAccess, CacheResult}

object Emit extends App {
  val p = Params()
  Files.createDirectories(Paths.get("build"))
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Preflight(p),
    firtoolOpts = args,
    args = Array("--target-dir", "build/Preflight", "--split-verilog")
  )

  val defs = scala.collection.mutable.ArrayBuffer[String](
    s"`define PF_ADDRESS_BITS ${p.bus.addressBits}",
    s"`define PF_BEAT_BYTES ${p.beatBytes}"
  )

  val ports = scala.collection.mutable.ArrayBuffer[String]()

  def channel(
    kind:   String,
    prefix: String,
    signal: String,
    bundle: Bundle
  ): Unit = {
    var offset = 0
    for ((name, field) <- bundle.elements.toSeq) {
      val width = field.getWidth
      defs += s"`define PF_${kind}_${name.toUpperCase}_OFFSET $offset"
      defs += s"`define PF_${kind}_${name.toUpperCase}_WIDTH $width"
      // firtool drops zero-width fields from the module ports.
      if (width > 0) ports += s".${prefix}_bits_$name($signal.bits[$offset +: $width])"
      offset += width
    }
    defs += s"`define PF_${kind}_WIDTH $offset"
    ports += s".${prefix}_valid($signal.valid)"; ports += s".${prefix}_ready($signal.ready)"
  }

  channel("CMD", "io_command", "cmd", new Command(p))
  channel("OUT", "io_prepared", "prepared", new PreparedSegment(p))
  channel("AUTH", "io_authorization", "authorization", new Authorization(p))
  channel("PERMIT", "io_permission", "permission", new Permission(p))
  channel("PTE", "io_pteRequest", "pte_req", new CacheAccess(p.bus))
  channel("PTERESP", "io_pteResponse", "pte_resp", new CacheResult)
  Files.writeString(Paths.get("build/preflight_config.svh"), defs.mkString("\n") + "\n")
  Files.writeString(Paths.get("build/preflight_ports.svh"), ports.mkString(",\n") + "\n")
  val lineRead       = p.copy(beatBytes = 64)
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Preflight(lineRead),
    firtoolOpts = args,
    args = Array("--target-dir", "build/Preflight64", "--split-verilog")
  )
  Files.writeString(
    Paths.get("build/preflight64_config.svh"),
    defs.map(d => if (d.startsWith("`define PF_BEAT_BYTES ")) s"`define PF_BEAT_BYTES ${lineRead.beatBytes}" else d)
      .mkString("\n") + "\n"
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new PreparedMap(p),
    firtoolOpts = args,
    args = Array("--target-dir", "build/PreparedMap", "--split-verilog")
  )
  ports.clear()
  channel("RESERVE", "io_reserve", "reserve", new MapTag(p))
  channel("MAP_READY", "io_ready", "mapped", new MapReady(p))
  channel("RELEASE", "io_release", "retire", new MapTag(p))
  // PreparedSegment uses the existing PF_OUT layout.
  var preparedOffset = 0
  for ((name, field)               <- (new PreparedSegment(p)).elements.toSeq) {
    ports += s".io_prepared_bits_$name(prepared.bits[$preparedOffset +: ${field.getWidth}])"
    preparedOffset += field.getWidth
  }
  ports += ".io_prepared_valid(prepared.valid)"; ports += ".io_prepared_ready(prepared.ready)"
  for (
    (kind, prefix, signal, bundle) <- Seq(
                                        ("QUERY", "io_queries", "query", new MapQuery(p)),
                                        ("RESULT", "io_results", "result", new MapResult(p))
                                      )
  ) {
    var offset = 0
    for ((name, field) <- bundle.elements.toSeq) {
      defs += s"`define PF_${kind}_${name.toUpperCase}_OFFSET $offset"
      defs += s"`define PF_${kind}_${name.toUpperCase}_WIDTH ${field.getWidth}"
      for (port <- 0 until 2) ports += s".${prefix}_${port}_$name(ctl.${signal}_$port[$offset +: ${field.getWidth}])"
      offset += field.getWidth
    }
    defs += s"`define PF_${kind}_WIDTH $offset"
  }
  Files.writeString(Paths.get("build/preflight_config.svh"), defs.mkString("\n") + "\n")
  Files.writeString(Paths.get("build/prepared_map_ports.svh"), ports.mkString(",\n") + "\n")
}
