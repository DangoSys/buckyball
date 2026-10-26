package framework.ip.bus.axi4

import java.nio.file.{Files, Paths}

object Emit extends App {
  val Array("--config", config, "--output-dir", output) = args
  val values                                            = ujson.read(Files.readString(Paths.get(config)))
  val p                                                 = Params(values("addressBits").num.toInt, values("dataBits").num.toInt, values("idBits").num.toInt)
  val masters                                           = values("masters").num.toInt
  val slots                                             = values("slots").num.toInt
  Files.createDirectories(Paths.get(output))
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Interconnect(InterconnectParam(p, masters, Some(slots))),
    args = Array("--target-dir", s"$output/Interconnect", "--split-verilog")
  )
  Files.writeString(
    Paths.get(output, "parameters.svh"),
    s"""`ifndef AXI_PARAMETERS
`define AXI_PARAMETERS
`define AXI_ADDRESS_BITS ${p.addressBits}
`define AXI_DATA_BITS ${p.dataBits}
`define AXI_ID_BITS ${p.idBits}
`define AXI_MASTERS $masters
`define AXI_SLOTS $slots
`endif
"""
  )

  val fields = Seq(
    "aw" -> Seq("id", "addr", "len", "size", "burst", "lock", "cache", "prot", "qos", "region"),
    "ar" -> Seq("id", "addr", "len", "size", "burst", "lock", "cache", "prot", "qos", "region"),
    "w"  -> Seq("data", "strb", "last"),
    "r"  -> Seq("id", "data", "resp", "last"),
    "b"  -> Seq("id", "resp")
  )

  val ports = (0 to masters).flatMap { index =>
    val prefix = if (index == masters) "io_out" else s"io_in_$index"
    fields.flatMap { case (channel, members) =>
      Seq(
        s".${prefix}_${channel}_valid(bus.ports[$index].${channel}_valid)",
        s".${prefix}_${channel}_ready(bus.ports[$index].${channel}_ready)"
      ) ++
        members.map(field => s".${prefix}_${channel}_bits_$field(bus.ports[$index].$channel.$field)")
    }
  }

  Files.writeString(Paths.get(output, "ports.svh"), ports.mkString(",\n"))
  val regions = values("regions").arr.toSeq.map(v => (BigInt(v("base").str), BigInt(v("bytes").str)))
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Decoder(regions, p),
    args = Array("--target-dir", s"$output/Decoder", "--split-verilog")
  )

  val decoderPorts = Seq("in", "matched", "other").zipWithIndex.flatMap { case (prefix, index) =>
    fields.flatMap { case (channel, members) =>
      Seq(
        s".io_${prefix}_${channel}_valid(bus.ports[$index].${channel}_valid)",
        s".io_${prefix}_${channel}_ready(bus.ports[$index].${channel}_ready)"
      ) ++ members.map(field => s".io_${prefix}_${channel}_bits_$field(bus.ports[$index].$channel.$field)")
    }
  }

  Files.writeString(Paths.get(output, "decoder_ports.svh"), decoderPorts.mkString(",\n"))
  Files.writeString(
    Paths.get(output, "decoder_regions.svh"),
    s"`define DECODER_REGIONS ${regions.size}\n" +
      s"`define DECODER_BASES ${regions.map { case (base, _) => s"${p.addressBits}'h${base.toString(16)}" }.mkString("'{", ",", "}")}\n" +
      s"`define DECODER_SIZES ${regions.map { case (_, size) => s"${p.addressBits}'h${size.toString(16)}" }.mkString("'{", ",", "}")}\n"
  )

}
