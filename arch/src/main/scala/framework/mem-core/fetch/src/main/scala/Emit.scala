package memcore.memory.fetch

import java.nio.file.{Files, Paths}

object Emit extends App {
  val p           = Params()
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Fetch(p),
    firtoolOpts = args,
    args = Array("--target-dir", "build/Fetch", "--split-verilog")
  )
  val definitions = scala.collection.mutable.ArrayBuffer[String]()
  val ports       = scala.collection.mutable.ArrayBuffer[String]()

  def fields(
    kind:     String,
    prefix:   String,
    instance: String,
    fs:       Seq[(String, Int)]
  ): Unit = {
    var offset = 0
    for ((name, width) <- fs) {
      definitions += s"`define FETCH_${kind}_${name.toUpperCase}_OFFSET $offset"
      definitions += s"`define FETCH_${kind}_${name.toUpperCase}_WIDTH $width"
      ports += s".$prefix$name($instance.bits[$offset +: $width])"
      offset += width
    }
    definitions += s"`define FETCH_${kind}_WIDTH $offset"
  }

  fields(
    "REQUEST",
    "io_request_bits_",
    "request",
    Seq(
      "addr"              -> 64,
      "context_privilege" -> 2,
      "context_satp"      -> 64,
      "context_sum"       -> 1,
      "context_mxr"       -> 1,
      "execute"           -> 1
    )
  )
  fields("RESPONSE", "io_response_bits_", "response", Seq("data" -> 64, "pageFault" -> 1, "accessFault" -> 1))
  fields(
    "PACKET",
    "io_packet_bits_",
    "sink_if",
    Seq("pc" -> 64, "data" -> 32, "mask" -> p.lanes, "pageFault" -> 1, "accessFault" -> 1)
  )
  for (
    (prefix, instance) <- Seq(
                            "request"     -> "request",
                            "response"    -> "response",
                            "packet"      -> "sink_if",
                            "maintenance" -> "maintenance",
                            "maintained"  -> "maintained"
                          )
  ) {
    ports += s".io_${prefix}_valid($instance.valid)"
    ports += s".io_${prefix}_ready($instance.ready)"
  }
  ports += ".io_maintenance_bits(maintenance.bits[0])"
  ports += ".io_maintained_bits(maintained.bits[0])"
  Files.write(Paths.get("build/fetch_config.svh"), (definitions.mkString("\n") + "\n").getBytes)
  Files.write(Paths.get("build/fetch_ports.svh"), ports.mkString(",\n").getBytes)
}
