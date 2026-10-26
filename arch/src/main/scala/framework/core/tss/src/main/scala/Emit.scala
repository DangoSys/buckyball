package framework.core.tss

import chisel3._
import java.nio.file.{Files, Paths}
import framework.ip.memory.spm
import framework.core.tss.configs.Params

object Emit extends App {
  val Array("--config", config, "--output-dir", output) = args
  val values                                            = ujson.read(Files.readString(Paths.get(config)))
  val mem                                               = values("memory")

  val p = Params(
    spm.Params(BigInt(mem("base").str), mem("bytes").num.toInt, mem("dataBits").num.toInt),
    values("ports").num.toInt
  )

  Files.createDirectories(Paths.get(output))
  val ports        = scala.collection.mutable.ArrayBuffer[String]()
  val definitions  = scala.collection.mutable.ArrayBuffer[String]()
  val offsets      = scala.collection.mutable.Map[String, Int]()
  var inputOffset  = 0
  var outputOffset = 0

  def fields(data: Data, path: String): Seq[(String, Int)] = data match {
    case record: Record => record.elements.toSeq.flatMap { case (name, field) =>
        fields(field, if (path.isEmpty) name else s"${path}_$name")
      }
    case vector: Vec[_] => vector.toSeq.zipWithIndex.flatMap { case (field, index) => fields(field, s"${path}_$index") }
    case leaf => Seq(path -> leaf.getWidth)
  }

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    {
      val dut = new Store(p)
      for ((name, width) <- fields(dut.io, "")) {
        val input     = name == "inUse" || name.startsWith("cancel_") || name.endsWith("request_valid") || name.contains(
          "request_bits_"
        ) || name.endsWith("response_ready")
        val direction = if (input) "IN" else "OUT"
        val signal    = if (input) "inputs" else "outputs"
        val offset    = if (input) inputOffset else outputOffset
        definitions += s"`define TSS_${direction}_${name.toUpperCase}_OFFSET $offset"
        definitions += s"`define TSS_${direction}_${name.toUpperCase}_WIDTH $width"
        ports += s".io_$name(bus.$signal[$offset +: $width])"
        offsets(name) = offset
        if (input) inputOffset += width else outputOffset += width
      }
      dut
    },
    args = Array("--target-dir", s"$output/Store", "--split-verilog")
  )
  definitions += s"`define TSS_INPUT_BITS $inputOffset"
  definitions += s"`define TSS_OUTPUT_BITS $outputOffset"
  definitions += s"`define TSS_CLIENTS ${p.ports}"
  definitions += s"`define TSS_PORTS ${p.ports + 1}"
  definitions += s"`define TSS_DATA_BITS ${p.memory.dataBits}"
  definitions += s"`define TSS_MASK_BITS ${p.memory.beatBytes}"
  definitions += s"`define TSS_SIZE_BITS ${p.memory.sizeBits}"
  definitions += s"`define TSS_BYTES ${p.memory.bytes}"
  definitions += s"`define TSS_BASE 64'h${p.memory.base.toString(16)}"

  val portOffsets = (0 to p.ports).map { index =>
    val name   = if (index == p.ports) "host" else s"clients_$index"
    val fields = Seq(
      "request_valid",
      "request_bits_address",
      "request_bits_size",
      "request_bits_write",
      "request_bits_data",
      "request_bits_mask",
      "request_ready",
      "response_valid",
      "response_bits_data",
      "response_bits_error",
      "response_ready"
    ).map(field => offsets(s"${name}_$field"))
    (fields ++ Seq(
      if (index == p.ports) offsets("hostBusy") else offsets(s"busy_$index"),
      if (index == p.ports) 0
      else offsets(s"cancel_$index")
    )).mkString("'{", ",", "}")
  }.mkString("'{", ",", "}")

  definitions += s"`define TSS_PORT_OFFSETS $portOffsets"
  Files.writeString(Paths.get(output, "parameters.svh"), definitions.mkString("\n") + "\n")
  Files.writeString(Paths.get(output, "ports.svh"), ports.mkString(",\n") + "\n")
}
