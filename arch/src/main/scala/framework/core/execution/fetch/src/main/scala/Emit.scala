package framework.core.execution.fetch

import chisel3._
import java.nio.file.{Files, Paths}
import framework.core.execution.fetch.configs.Params

object Emit extends App {
  val Array("--config", config, "--output-dir", output) = args
  val p                                                 = upickle.default.read[Params](Files.readString(Paths.get(config)))
  Files.createDirectories(Paths.get(output))
  val ports                                             = scala.collection.mutable.ArrayBuffer[String]()
  val definitions                                       = scala.collection.mutable.ArrayBuffer[String]()
  val offsets                                           = scala.collection.mutable.Map[String, Int]()
  val widths                                            = scala.collection.mutable.Map[String, Int]()
  var inputOffset                                       = 0
  var outputOffset                                      = 0

  def fields(data: Data, path: String): Seq[(String, Int)] = data match {
    case record: Record => record.elements.toSeq.flatMap { case (name, field) =>
        fields(field, if (path.isEmpty) name else s"${path}_$name")
      }
    case vector: Vec[_] => vector.toSeq.zipWithIndex.flatMap { case (field, index) => fields(field, s"${path}_$index") }
    case leaf => Seq(path -> leaf.getWidth)
  }

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    {
      val dut = new Fetch(p)
      for ((name, width) <- fields(dut.io, "")) {
        val input     = Set(
          "resetVector",
          "flush",
          "invalidate",
          "redirect_valid",
          "redirect_bits",
          "packet_ready",
          "request_ready",
          "response_valid",
          "maintenance_ready",
          "maintained_valid",
          "maintained_bits"
        ).contains(name) || name.startsWith("context_") || name.startsWith("response_bits_")
        val direction = if (input) "IN" else "OUT"
        val signal    = if (input) "inputs" else "outputs"
        val offset    = if (input) inputOffset else outputOffset
        definitions += s"`define FETCH_${direction}_${name.toUpperCase}_OFFSET $offset"
        definitions += s"`define FETCH_${direction}_${name.toUpperCase}_WIDTH $width"
        ports += s".io_$name(bus.$signal[$offset +: $width])"
        offsets(name) = offset; widths(name) = width
        if (input) inputOffset += width else outputOffset += width
      }
      dut
    },
    args = Array("--target-dir", s"$output/Fetch", "--split-verilog")
  )
  definitions += s"`define FETCH_INPUT_BITS $inputOffset"
  definitions += s"`define FETCH_OUTPUT_BITS $outputOffset"
  definitions += s"`define FETCH_INSTRUCTION_BYTES ${p.instructionBytes}"
  definitions += s"`define FETCH_MASK_VALUES ${if (p.instructionBytes == 2) "{2,3}" else "{1}"}"
  for (
    (name, prefix) <- Seq("REQUEST" -> "request_bits_", "RESPONSE" -> "response_bits_", "PACKET" -> "packet_bits_")
  ) {
    val names = offsets.keys.filter(_.startsWith(prefix)).toSeq
    definitions += s"`define FETCH_${name}_OFFSET ${names.map(offsets).min}"
    definitions += s"`define FETCH_${name}_BITS ${names.map(widths).sum}"
  }
  Files.writeString(Paths.get(output, "parameters.svh"), definitions.mkString("\n") + "\n")
  Files.writeString(Paths.get(output, "ports.svh"), ports.mkString(",\n") + "\n")
}
