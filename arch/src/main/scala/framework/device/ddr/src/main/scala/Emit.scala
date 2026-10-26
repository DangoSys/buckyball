package framework.device.ddr

import chisel3._
import java.nio.file.{Files, Paths}
import framework.ip.bus.axi4
import framework.ip.bus.chi
import framework.ip.bus.chi.snf.{LineRequest, LineResponse}

object Emit extends App {
  val Array("--config", config, "--output-dir", directory) = args
  val values                                               = ujson.read(Files.readString(Paths.get(config)))

  val p = Params(
    upickle.default.read[chi.Params](values("line")),
    values("clients").num.toInt,
    values("slotsPerClient").num.toInt,
    values("dataBits").num.toInt,
    values("idBits").num.toInt,
    values("idBase").num.toInt,
    BigInt(values("base").str),
    BigInt(values("bytes").str),
    BigInt(values("model").str)
  )

  Files.createDirectories(Paths.get(directory))
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Bridge(p),
    args = Array("--target-dir", s"$directory/Bridge", "--split-verilog")
  )

  val defs = scala.collection.mutable.ArrayBuffer[String](
    s"`define DDR_DATA_BITS ${p.dataBits}",
    s"`define DDR_BEAT_BYTES ${p.axi.bytes}",
    s"`define DDR_BEATS ${p.beats}",
    s"`define DDR_SLOTS ${p.slots}",
    s"`define DDR_SLOTS_PER_CLIENT ${p.slotsPerClient}",
    s"`define DDR_ID_BASE ${p.idBase}"
  )

  val ports = scala.collection.mutable.ArrayBuffer[String]()

  def channel(
    kind:   String,
    prefix: String,
    signal: String,
    bundle: Bundle,
    define: Boolean = true
  ): Unit = {
    var offset = 0
    for ((name, field) <- bundle.elements.toSeq) {
      val width = field.getWidth
      if (define) {
        defs += s"`define DDR_${kind}_${name.toUpperCase}_OFFSET $offset"
        defs += s"`define DDR_${kind}_${name.toUpperCase}_WIDTH $width"
      }
      ports += s".${prefix}_bits_$name($signal.bits[$offset +: $width])"
      offset += width
    }
    if (define) defs += s"`define DDR_${kind}_WIDTH $offset"
    ports += s".${prefix}_valid($signal.valid)"
    ports += s".${prefix}_ready($signal.ready)"
  }

  for (c <- 0 until p.clients) {
    channel(
      "REQ",
      s"io_request_$c",
      s"req$c",
      new LineRequest(p.line),
      c == 0
    )
    channel(
      "RESP",
      s"io_response_$c",
      s"resp$c",
      new LineResponse(p.line),
      c == 0
    )
  }
  channel(
    "A",
    "io_axi_aw",
    "aw",
    new axi4.Address(p.axi)
  )
  channel(
    "A",
    "io_axi_ar",
    "ar",
    new axi4.Address(p.axi),
    false
  )
  channel(
    "W",
    "io_axi_w",
    "w",
    new axi4.WriteData(p.axi)
  )
  channel(
    "B",
    "io_axi_b",
    "b",
    new axi4.WriteResponse(p.axi)
  )
  channel(
    "R",
    "io_axi_r",
    "r",
    new axi4.ReadData(p.axi)
  )
  Files.writeString(
    Paths.get(s"$directory/parameters.svh"),
    "`ifndef DDR_CONFIG_SVH\n`define DDR_CONFIG_SVH\n" + defs.mkString("\n") + "\n`endif\n"
  )
  Files.writeString(Paths.get(s"$directory/ports.svh"), ports.mkString(",\n") + "\n")
}
