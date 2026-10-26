package framework.acc.memdomain.tomem

import chisel3._
import chisel3.reflect.DataMirror
import java.nio.file.{Files, Paths}
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.acc.gpdomain.configs.GpDomainParam
import framework.core.memory.access.preflight.configs.PmpParams

object Emit extends App {
  val Array("--config", config, "--output-dir", directory) = args
  val values                                               = ujson.read(Files.readString(Paths.get(config)))
  val frontend                                             = upickle.default.read[FrontendParam](values("frontend"))
  val memory                                               = upickle.default.read[MemDomainParam](values("memory"))
  val gp                                                   = upickle.default.read[GpDomainParam](values("gp"))
  val permission                                           = upickle.default.read[PmpParams](values("permission"))
  val transport                                            = upickle.default.read[configs.Params](values("transport"))

  val definitions = scala.collection.mutable.ArrayBuffer[String](
    s"`define TM_BANKS ${memory.virtualBankCount + 2}",
    s"`define TM_ROWS ${memory.bankEntries}",
    s"`define TM_GROUPS ${memory.bankNum}",
    s"`define TM_DATA_BITS ${memory.bankWidth}"
  )

  val binding     = scala.collection.mutable.ArrayBuffer[String]()
  var inputOffset = 0; var outputOffset = 0
  Files.createDirectories(Paths.get(directory))

  def leaves(data: Data, path: String): Seq[(String, Data)] = data match {
    case record: Record => record.elements.toSeq.flatMap { case (name, field) =>
        leaves(field, if (path.isEmpty) name else s"${path}_$name")
      }
    case vector: Vec[_] => vector.toSeq.zipWithIndex.flatMap { case (field, index) => leaves(field, s"${path}_$index") }
    case field => if (field.getWidth > 0) Seq(path -> field) else Seq.empty
  }

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    {
      val dut = new ToMem(frontend, memory, gp, transport, permission)
      for ((name, field) <- leaves(dut.io, "")) {
        val input     = DataMirror.directionOf(field) == ActualDirection.Input
        val direction = if (input) "IN" else "OUT"; val signal              = if (input) "inputs" else "outputs";
        val offset    = if (input) inputOffset else outputOffset; val width = field.getWidth
        definitions += s"`define TM_${direction}_${name.toUpperCase}_OFFSET $offset";
        definitions += s"`define TM_${direction}_${name.toUpperCase}_WIDTH $width"
        binding += s".io_$name(bus.$signal[$offset +: $width])";
        if (input) inputOffset += width else outputOffset += width
      }; dut
    },
    args = Array("--target-dir", s"$directory/ToMem", "--split-verilog")
  )
  definitions += s"`define TM_INPUT_BITS $inputOffset"; definitions += s"`define TM_OUTPUT_BITS $outputOffset"
  Files.writeString(Paths.get(directory, "parameters.svh"), definitions.mkString("\n") + "\n")
  Files.writeString(Paths.get(directory, "ports.svh"), binding.mkString(",\n") + "\n")
}
