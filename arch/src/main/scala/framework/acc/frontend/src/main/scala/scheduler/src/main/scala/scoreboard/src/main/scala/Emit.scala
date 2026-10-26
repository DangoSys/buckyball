package framework.acc.frontend.scheduler.scoreboard

import chisel3._
import java.nio.file.{Files, Paths}
import framework.acc.frontend.scheduler.scoreboard.configs.Params

object Emit extends App {
  val Array("--config", config, "--output-dir", output) = args
  val p                                                 = upickle.default.read[Params](Files.readString(Paths.get(config)))
  Files.createDirectories(Paths.get(output))

  def fields(data: Data, path: String): Seq[(String, Int)] = data match {
    case record: Record => record.elements.toSeq.flatMap { case (name, field) =>
        fields(field, if (path.isEmpty) name else s"${path}_$name")
      }
    case vector: Vec[_] => vector.toSeq.zipWithIndex.flatMap { case (field, index) => fields(field, s"${path}_$index") }
    case leaf => Seq(path -> leaf.getWidth)
  }

  val inputRoots   = Set("alloc", "issue", "complete", "queryVec")
  val ports        = scala.collection.mutable.ArrayBuffer[String]()
  val definitions  = scala.collection.mutable.ArrayBuffer[String]()
  val offsets      = scala.collection.mutable.Map[String, Int]()
  var inputOffset  = 0
  var outputOffset = 0
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    {
      val dut = new BankScoreboard(p)
      for {
        (root, bundle) <- dut.io.elements.toSeq
        (name, width)  <- fields(bundle, root)
      } {
        val input     = inputRoots.contains(root)
        val direction = if (input) "IN" else "OUT"
        val signal    = if (input) "inputs" else "outputs"
        val offset    = if (input) inputOffset else outputOffset
        definitions += s"`define SB_${direction}_${name.toUpperCase}_OFFSET $offset"
        definitions += s"`define SB_${direction}_${name.toUpperCase}_WIDTH $width"
        ports += s".io_$name(bus.$signal[$offset +: $width])"
        offsets(name) = offset
        if (input) inputOffset += width else outputOffset += width
      }
      dut
    },
    args = Array("--target-dir", s"$output/BankScoreboard", "--split-verilog")
  )
  definitions += s"`define SB_INPUT_BITS $inputOffset"
  definitions += s"`define SB_OUTPUT_BITS $outputOffset"
  definitions += s"`define SB_BANKIDBITS ${p.bankIdBits}"
  definitions += s"`define SB_BANKS ${p.banks}"
  definitions += s"`define SB_ENTRIES ${p.entries}"

  val queryOffsets = (0 until p.entries).map { index =>
    Seq("rd_bank_0_valid", "rd_bank_0_id", "rd_bank_1_valid", "rd_bank_1_id", "rd_wr_valid", "rd_wr_id")
      .map(name => offsets(s"queryVec_${index}_$name")).mkString("'{", ",", "}")
  }.mkString("'{", ",", "}")

  definitions += s"`define SB_QUERY_OFFSETS $queryOffsets"
  Files.writeString(Paths.get(output, "parameters.svh"), definitions.mkString("\n") + "\n")
  Files.writeString(Paths.get(output, "ports.svh"), ports.mkString(",\n") + "\n")
}
