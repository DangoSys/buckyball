package framework.acc.frontend.boot

import chisel3._
import java.nio.file.{Files, Paths}
import framework.acc.frontend.configs.FrontendParam
import framework.core.memory.access.preflight.configs.PmpParams

object Emit extends App {
  val Array("--config", config, "--output-dir", output) = args
  val values                                            = ujson.read(Files.readString(Paths.get(config)))
  val frontend                                          = upickle.default.read[FrontendParam](values("frontend"))
  val permission                                        = upickle.default.read[PmpParams](values("permission"))
  val ballIds                                           = values("ballIds").arr.toSeq.map(_.num.toInt)
  Files.createDirectories(Paths.get(output))
  val ports                                             = scala.collection.mutable.ArrayBuffer[String]()
  val definitions                                       = scala.collection.mutable.ArrayBuffer[String]()
  val offsets                                           = scala.collection.mutable.Map[String, Int]()
  var inputOffset                                       = 0
  var outputOffset                                      = 0
  var packetBits                                        = 0

  def fields(data: Data, path: String): Seq[(String, Int)] = data match {
    case record: Record => record.elements.toSeq.flatMap { case (name, field) =>
        fields(field, if (path.isEmpty) name else s"${path}_$name")
      }
    case vector: Vec[_] => vector.toSeq.zipWithIndex.flatMap { case (field, index) => fields(field, s"${path}_$index") }
    case leaf => Seq(path -> leaf.getWidth)
  }

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    {
      val dut = new BootRom(ballIds, permission, frontend)
      packetBits = dut.io.command.bits.getWidth
      for ((name, width) <- fields(dut.io, "")) {
        val input     =
          name == "cmd_ready" || name == "schedulerIdle" || name == "command_valid" || name.startsWith("command_bits_")
        val direction = if (input) "IN" else "OUT"
        val signal    = if (input) "inputs" else "outputs"
        val offset    = if (input) inputOffset else outputOffset
        definitions += s"`define BOOT_${direction}_${name.toUpperCase}_OFFSET $offset"
        definitions += s"`define BOOT_${direction}_${name.toUpperCase}_WIDTH $width"
        offsets(name) = offset
        ports += s".io_$name(bus.$signal[$offset +: $width])"
        if (input) inputOffset += width else outputOffset += width
      }
      dut
    },
    args = Array("--target-dir", s"$output/BootRom", "--split-verilog")
  )
  definitions += s"`define BOOT_INPUT_BITS $inputOffset"
  definitions += s"`define BOOT_OUTPUT_BITS $outputOffset"
  definitions += s"`define BOOT_PACKET_BITS $packetBits"
  definitions += s"`define BOOT_IN_PACKET_OFFSET ${offsets.filter(_._1.startsWith("command_bits_")).values.min}"
  definitions += s"`define BOOT_OUT_PACKET_OFFSET ${offsets.filter(_._1.startsWith("cmd_bits_")).values.min}"
  definitions += s"`define BOOT_BALL_COUNT ${ballIds.size}"
  definitions += s"`define BOOT_BALL_IDS ${ballIds.mkString("'{", ",", "}")}"
  definitions += s"`define BOOT_ID_BINS ${ballIds.mkString("{", ",", "}")}"
  Files.writeString(Paths.get(output, "parameters.svh"), definitions.mkString("\n") + "\n")
  Files.writeString(Paths.get(output, "ports.svh"), ports.mkString(",\n") + "\n")
}
