package framework.core.memory.access.mmu

import chisel3._
import java.nio.file.{Files, Paths}
import framework.core.memory.access.mmu.configs.Params

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
      val dut = new Walker(p)
      for ((name, width) <- fields(dut.io, "")) {
        val input     = name.startsWith("config_") || name == "flush" || name.startsWith("lookup_config_") ||
          name.startsWith("lookup_req_") || name == "req_valid" || name.startsWith("req_bits_") ||
          name == "resp_ready" || name == "access_ready" || name == "result_valid" || name.startsWith("result_bits_")
        val direction = if (input) "IN" else "OUT"
        val signal    = if (input) "inputs" else "outputs"
        val offset    = if (input) inputOffset else outputOffset
        definitions += s"`define MMU_${direction}_${name.toUpperCase}_OFFSET $offset"
        definitions += s"`define MMU_${direction}_${name.toUpperCase}_WIDTH $width"
        ports += s".io_$name(bus.$signal[$offset +: $width])"
        offsets(name) = offset; widths(name) = width
        if (input) inputOffset += width else outputOffset += width
      }
      dut
    },
    args = Array("--target-dir", s"$output/Walker", "--split-verilog")
  )
  definitions += s"`define MMU_INPUT_BITS $inputOffset"
  definitions += s"`define MMU_OUTPUT_BITS $outputOffset"
  definitions += s"`define MMU_TLB_ENTRIES ${p.entries}"
  definitions += s"`define MMU_ADDRESS_BITS ${p.bus.addressBits}"
  for ((name, prefix) <- Seq("REQ" -> "req_bits_", "RESP" -> "resp_bits_", "LOOKUP" -> "lookup_req_")) {
    val names = offsets.keys.filter(_.startsWith(prefix)).toSeq
    definitions += s"`define MMU_${name}_OFFSET ${names.map(offsets).min}"
    definitions += s"`define MMU_${name}_BITS ${names.map(widths).sum}"
  }
  Files.writeString(Paths.get(output, "parameters.svh"), definitions.mkString("\n") + "\n")
  Files.writeString(Paths.get(output, "ports.svh"), ports.mkString(",\n") + "\n")
}
