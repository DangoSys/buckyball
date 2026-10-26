package framework.core.controller.admission

import chisel3._
import chisel3.reflect.DataMirror
import java.nio.file.{Files, Paths}
import framework.core.controller.configs.CommandsParam
import framework.core.memory.access.preflight.configs.PmpParams
import framework.ip.memory.interlock.{Params => TrackingParam}

object Emit extends App {
  val Array("--config", config, "--output-dir", directory) = args
  val values                                               = ujson.read(Files.readString(Paths.get(config)))

  implicit val trackingRW: upickle.default.ReadWriter[TrackingParam] = upickle.default.macroRW

  val p = CommandsParam(
    upickle.default.read[TrackingParam](values("tracking")),
    values("dataBits").num.toInt,
    upickle.default.read[PmpParams](values("permission")),
    values("moves").bool,
    values("taskEnabled").bool,
    values("remote").bool
  )

  Files.createDirectories(Paths.get(directory))

  def leaves(data: Data, path: String): Seq[(String, Data)] = data match {
    case record: Record => record.elements.toSeq.flatMap { case (name, field) =>
        leaves(field, if (path.isEmpty) name else s"${path}_$name")
      }
    case vector: Vec[_] => vector.toSeq.zipWithIndex.flatMap { case (field, index) => leaves(field, s"${path}_$index") }
    case field => if (field.getWidth > 0) Seq(path -> field) else Seq.empty
  }

  for ((top, kind) <- Seq("CommandAdmission", "MoveAdmission", "DirectAdmission", "TaskRouter").zipWithIndex) {
    val definitions = scala.collection.mutable.ArrayBuffer[String](
      s"`define AD_KIND $kind",
      s"`define AD_ENTRIES ${p.tracking.entries}",
      s"`define AD_DATA_BITS ${p.dataBits}"
    )
    val binding     = scala.collection.mutable.ArrayBuffer[String]()
    val offsets     = scala.collection.mutable.Map[String, Int]()
    val widths      = scala.collection.mutable.Map[String, Int]()
    var inputOffset = 0; var outputOffset = 0
    _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
      {
        val (dut, io) = kind match {
          case 0 => val m = new CommandAdmission(p); (m, m.io)
          case 1 => val m = new MoveAdmission(p.dataBits, p.permission); (m, m.io)
          case 2 => val m = new DirectAdmission(p); (m, m.io)
          case 3 => val m = new TaskRouter(p); (m, m.io)
        }
        for ((name, field) <- leaves(io, "")) {
          val input     = DataMirror.directionOf(field) == ActualDirection.Input
          val direction = if (input) "IN" else "OUT"; val signal              = if (input) "inputs" else "outputs";
          val offset    = if (input) inputOffset else outputOffset; val width = field.getWidth
          definitions += s"`define AD_${direction}_${name.toUpperCase}_OFFSET $offset";
          definitions += s"`define AD_${direction}_${name.toUpperCase}_WIDTH $width"
          offsets(name) = offset; widths(name) = width
          binding += s".io_$name(bus.$signal[$offset +: $width])";
          if (input) inputOffset += width else outputOffset += width
        }; dut
      },
      args = Array("--target-dir", s"$directory/$top", "--split-verilog")
    )
    val command     = kind match { case 1 => "rocc_cmd"; case 3 => "command"; case _ => "execution_command" }
    val source      = s"${command}_bits_"
    val instruction = s"${source}instruction_"
    val maps        = scala.collection.mutable.ArrayBuffer[(Int, Int, Int, Int)]()
    def copyPayload(target: String): Unit = for (name <- offsets.keys.toSeq.sorted if name.startsWith(target)) {
      val from = source + name.stripPrefix(target);
      maps += ((offsets(from), offsets(name), widths(name), offsets(name.take(name.indexOf("_bits")) + "_valid")))
    }
    def control(target: String): Unit = {
      for (
        (name, from, lsb) <-
          Seq(("operation", "raw_inst", 25), ("context", "rs1Data", 32), ("field", "rs1Data", 0), ("data", "rs2Data", 0))
      ) {
        val to = s"${target}_$name";
        maps += (
          (
            offsets(instruction + from) + lsb,
            offsets(to),
            widths(to),
            offsets(to.take(to.indexOf("_bits")) + "_valid")
          )
        )
      }
    }
    kind match {
      case 0 => copyPayload("taskRequest_bits_")
      case 1 => ()
      case 2 => copyPayload("accelerator_cmd_bits_"); control("remote_request_bits")
      case 3 =>
        control("local_request_bits_control"); control("remote_request_bits")
        for (name <- offsets.keys.toSeq.sorted if name.startsWith("local_request_bits_context_")) {
          val from = source + name.stripPrefix("local_request_bits_context_");
          maps += ((offsets(from), offsets(name), widths(name), offsets(name.take(name.indexOf("_bits")) + "_valid")))
        }
    }
    if (kind == 0 || kind == 1) {
      for (
        (name, from, lsb) <- Seq(
                               ("sourceThread", "rs1Data", 0),
                               ("targetThread", "rs1Data", 8),
                               ("sourceBank", "rs1Data", 16),
                               ("targetBank", "rs1Data", 26),
                               ("sourceAddr", "rs2Data", 0),
                               ("targetAddr", "rs2Data", 16)
                             )
      ) {
        val to = s"move_command_bits_$name";
        maps += (
          (
            offsets(instruction + from) + lsb,
            offsets(to),
            widths(to),
            offsets(to.take(to.indexOf("_bits")) + "_valid")
          )
        )
      }
    }
    definitions += s"`define AD_PAYLOAD_MAP ${maps.map { case (from, to, width, gate) =>
      s"'{$from,$to,$width,$gate}"
    }.mkString("'{", ",", "}")}"
    definitions += s"`define AD_PAYLOAD_FIELDS ${maps.size}"
    val commandFields = offsets.keys.filter(_.startsWith(source)).toSeq
    definitions += s"`define AD_COMMAND_OFFSET ${commandFields.map(offsets).min}"
    definitions += s"`define AD_COMMAND_WIDTH ${commandFields.map(widths).sum}"
    definitions += s"`define AD_INPUT_BITS $inputOffset"; definitions += s"`define AD_OUTPUT_BITS $outputOffset"
    Files.writeString(Paths.get(directory, top, "parameters.svh"), definitions.mkString("\n") + "\n")
    Files.writeString(Paths.get(directory, top, "ports.svh"), binding.mkString(",\n") + "\n")
  }
}
