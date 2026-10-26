package framework.acc.memdomain.tol1

import framework.ip.memory.interlock._

import chisel3._
import java.nio.file.{Files, Paths}

object Emit extends App {
  val Array("--config", config, "--output-dir", output) = args
  val values                                            = ujson.read(Files.readString(Paths.get(config)))

  val p = Params(
    values("entries").num.toInt,
    values("addressBits").num.toInt,
    values("idBits").num.toInt,
    values("lineBytes").num.toInt,
    values("maxRanges").num.toInt
  )

  Files.createDirectories(Paths.get(output))
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new ToL1(p),
    args = Array("--target-dir", s"$output/ToL1", "--split-verilog")
  )

  val definitions = scala.collection.mutable.ArrayBuffer[String](
    s"`define INTERLOCK_ENTRIES ${p.entries}",
    s"`define INTERLOCK_ADDRESS_BITS ${p.addressBits}",
    s"`define INTERLOCK_ID_BITS ${p.idBits}",
    s"`define INTERLOCK_LINE_BYTES ${p.lineBytes}",
    s"`define INTERLOCK_MAX_RANGES ${p.maxRanges}"
  )

  val ports = scala.collection.mutable.ArrayBuffer[String]()

  def channel(
    kind:      String,
    prefix:    String,
    signal:    String,
    data:      Bundle,
    decoupled: Boolean = true
  ): Unit = {
    var offset = 0
    for ((name, field) <- data.elements.toSeq) {
      val width = field.getWidth
      definitions += s"`define INTERLOCK_${kind}_${name.toUpperCase}_OFFSET $offset"
      definitions += s"`define INTERLOCK_${kind}_${name.toUpperCase}_WIDTH $width"
      ports += s".$prefix$name(bus.signals.${signal}_bits[$offset +: $width])"
      offset += width
    }
    definitions += s"`define INTERLOCK_${kind}_WIDTH $offset"
    if (decoupled) {
      val channelPrefix = prefix.stripSuffix("bits_")
      ports += s".${channelPrefix}valid(bus.signals.${signal}_valid)"
      ports += s".${channelPrefix}ready(bus.signals.${signal}_ready)"
    }
  }

  channel("DISPATCH", "io_dispatch_bits_", "dispatch", new Dispatch(p))
  channel("CANCEL", "io_cancel_bits_", "cancel", new Dispatch(p))
  channel("ACCESS_INFO", "io_accessInfo_bits_", "access_info", new AccessInfo(p))
  channel("MAINTENANCE", "io_maintenance_bits_", "maintenance", new Maintenance(p))
  channel("MAINTAINED", "io_maintained_bits_", "maintained", new Acknowledgement(p))
  channel("GRANT", "io_grant_bits_", "grant", new Tag(p))
  channel("DONE", "io_done_bits_", "done", new Acknowledgement(p))
  channel("COMPLETE", "io_complete_bits_", "complete", new Tag(p))
  channel("CPU_QUERY", "io_cpuQuery_", "cpu_query", new CpuQuery(p), decoupled = false)
  ports += ".io_cpuAllow(bus.signals.cpu_allow)"
  ports += ".io_cpuProbeAllow(bus.signals.cpu_probe_allow)"
  Files.write(Paths.get(output, "parameters.svh"), (definitions.mkString("\n") + "\n").getBytes)
  Files.write(Paths.get(output, "ports.svh"), ports.mkString(",\n").getBytes)
}
