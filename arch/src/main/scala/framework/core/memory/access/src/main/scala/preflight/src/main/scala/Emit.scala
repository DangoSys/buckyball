package framework.core.memory.access.preflight

import chisel3._
import java.nio.file.{Files, Paths}
import framework.ip.bus.chi
import framework.core.memory.access.mmu.{PteRequest, PteResponse}

object Emit extends App {
  val Array("--config", config, "--output-dir", directory) = args
  val values                                               = ujson.read(Files.readString(Paths.get(config)))
  val bus                                                  = upickle.default.read[chi.Params](values("bus"))(upickle.default.macroRW[chi.Params])

  val p = Params(
    bus,
    values("contexts").num.toInt,
    values("maxRanges").num.toInt,
    values("idBits").num.toInt,
    values("beatBytes").num.toInt
  )

  Files.createDirectories(Paths.get(directory))
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Preflight(p),
    args = Array("--target-dir", s"$directory/Preflight", "--split-verilog")
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new PreparedMap(p),
    args = Array("--target-dir", s"$directory/PreparedMap", "--split-verilog")
  )

  val defs = scala.collection.mutable.ArrayBuffer[String](
    s"`define PF_ADDRESS_BITS ${bus.addressBits}",
    s"`define PF_BEAT_BYTES ${p.beatBytes}",
    s"`define PF_CONTEXTS ${p.contexts}"
  )

  val ports    = scala.collection.mutable.ArrayBuffer[String]()
  val mapPorts = scala.collection.mutable.ArrayBuffer[String]()

  def channel(
    kind:   String,
    prefix: String,
    signal: String,
    bundle: Bundle,
    target: scala.collection.mutable.ArrayBuffer[String],
    stream: Boolean = true,
    define: Boolean = true
  ): Unit = {
    var offset = 0
    for ((name, field) <- bundle.elements.toSeq) {
      val width   = field.getWidth
      if (define) {
        defs += s"`define PF_${kind}_${name.toUpperCase}_OFFSET $offset";
        defs += s"`define PF_${kind}_${name.toUpperCase}_WIDTH $width"
      }
      val bits    = if (stream) "_bits" else ""
      val payload = if (stream) s"$signal.bits" else signal
      target += s".$prefix${bits}_$name($payload[$offset +: $width])"; offset += width
    }
    if (define) defs += s"`define PF_${kind}_WIDTH $offset"
    if (stream) { target += s".${prefix}_valid($signal.valid)"; target += s".${prefix}_ready($signal.ready)" }
  }

  channel("CMD", "io_command", "cmd", new Command(p), ports)
  channel("OUT", "io_prepared", "prepared", new PreparedSegment(p), ports)
  channel("AUTH", "io_authorization", "authorization", new Authorization(p), ports)
  channel("PERMIT", "io_permission", "permission", new Permission(p), ports)
  channel("PTE", "io_pteRequest", "pte_req", new PteRequest(p.walker), ports)
  channel("PTERESP", "io_pteResponse", "pte_resp", new PteResponse, ports)
  channel("RESERVE", "io_reserve", "reserve", new MapTag(p), mapPorts)
  channel("OUT", "io_prepared", "prepared", new PreparedSegment(p), mapPorts, define = false)
  channel("MAP_READY", "io_ready", "mapped", new MapReady(p), mapPorts)
  channel("RELEASE", "io_release", "retire", new MapTag(p), mapPorts)
  for (i <- 0 until 2) {
    channel("QUERY", s"io_queries_$i", s"ctl.query_$i", new MapQuery(p), mapPorts, stream = false, define = i == 0)
    channel("RESULT", s"io_results_$i", s"ctl.result_$i", new MapResult(p), mapPorts, stream = false, define = i == 0)
  }
  Files.writeString(Paths.get(directory, "parameters.svh"), defs.mkString("\n") + "\n")
  Files.writeString(Paths.get(directory, "preflight_ports.svh"), ports.mkString(",\n") + "\n")
  Files.writeString(Paths.get(directory, "prepared_map_ports.svh"), mapPorts.mkString(",\n") + "\n")
  val permission = upickle.default.read[configs.PmpParams](values("permission"))

  val regions = values("regions").arr.toSeq.map(v =>
    upickle.default.read[framework.core.memory.access.PhysicalRegion](v)(configs.Config.regionRW)
  )

  val spec = configs.Config(permission, p, regions)

  val definitions = scala.collection.mutable.ArrayBuffer[String](
    s"`define PREP_CONTEXTS ${p.contexts}",
    s"`define PREP_ID_BITS ${p.idBits}",
    s"`define PREP_ADDRESS_BITS ${bus.addressBits}",
    s"`define PREP_PMP_ENTRIES ${permission.entries}"
  )

  val binding     = scala.collection.mutable.ArrayBuffer[String]()
  var inputOffset = 0; var outputOffset = 0

  def leaves(data: Data, path: String): Seq[(String, Data)] = data match {
    case record: Record => record.elements.toSeq.flatMap { case (name, field) =>
        leaves(field, if (path.isEmpty) name else s"${path}_$name")
      }
    case vector: Vec[_] => vector.toSeq.zipWithIndex.flatMap { case (field, index) => leaves(field, s"${path}_$index") }
    case field => Seq(path -> field)
  }

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    {
      val dut = new Preparation(spec, p.walker)
      for ((name, field) <- leaves(dut.io, "")) {
        val input     = chisel3.reflect.DataMirror.directionOf(field) == ActualDirection.Input
        val direction = if (input) "IN" else "OUT"; val signal              = if (input) "inputs" else "outputs";
        val offset    = if (input) inputOffset else outputOffset; val width = field.getWidth
        definitions += s"`define PREP_${direction}_${name.toUpperCase}_OFFSET $offset";
        definitions += s"`define PREP_${direction}_${name.toUpperCase}_WIDTH $width"
        binding += s".io_$name(bus.$signal[$offset +: $width])";
        if (input) inputOffset += width else outputOffset += width
      }; dut
    },
    args = Array("--target-dir", s"$directory/Preparation", "--split-verilog")
  )
  definitions += s"`define PREP_INPUT_BITS $inputOffset"; definitions += s"`define PREP_OUTPUT_BITS $outputOffset"
  Files.writeString(Paths.get(directory, "preparation_parameters.svh"), definitions.mkString("\n") + "\n")
  Files.writeString(Paths.get(directory, "preparation_ports.svh"), binding.mkString(",\n") + "\n")

}
