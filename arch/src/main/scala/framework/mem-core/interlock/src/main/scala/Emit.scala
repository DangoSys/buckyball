package memcore.memory.interlock

import chisel3._
import java.nio.file.{Files, Paths}

object Emit extends App {
  val p = Params()
  Files.createDirectories(Paths.get("build"))
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Interlock(p),
    firtoolOpts = args,
    args = Array("--target-dir", "build/Interlock", "--split-verilog")
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
      ports += s".$prefix$name($signal.bits[$offset +: $width])"
      offset += width
    }
    definitions += s"`define INTERLOCK_${kind}_WIDTH $offset"
    if (decoupled) {
      val channelPrefix = prefix.stripSuffix("bits_")
      ports += s".${channelPrefix}valid($signal.valid)"
      ports += s".${channelPrefix}ready($signal.ready)"
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
  ports += ".io_cpuAllow(cpu_allow)"
  Files.write(Paths.get("build/interlock_config.svh"), (definitions.mkString("\n") + "\n").getBytes)
  Files.write(Paths.get("build/interlock_ports.svh"), ports.mkString(",\n").getBytes)
}
