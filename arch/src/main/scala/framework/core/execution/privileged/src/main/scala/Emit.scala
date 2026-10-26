package framework.core.execution.privileged

import chisel3._
import java.nio.file.{Files, Paths}
import framework.core.execution.privileged.configs.Params

object Emit extends App {
  val Array("--config", config, "--output-dir", output) = args
  val p                                                 = upickle.default.read[Params](Files.readString(Paths.get(config)))
  Files.createDirectories(Paths.get(output))
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Privileged(p),
    args = Array("--target-dir", s"$output/Privileged", "--split-verilog")
  )
  val inputs                                            = Set("hartId", "interrupts", "trap", "returning", "retired", "floating", "authorization")

  def fields(data: Data, path: String): Seq[(String, Int, Boolean)] = data match {
    case record: Record => record.elements.toSeq.flatMap { case (name, field) =>
        fields(field, if (path.isEmpty) name else s"${path}_$name")
      }
    case vector: Vec[_] => vector.toSeq.zipWithIndex.flatMap { case (field, index) => fields(field, s"${path}_$index") }
    case leaf => Seq((
        path,
        leaf.getWidth,
        inputs.contains(path.takeWhile(_ != '_')) ||
          (path.startsWith("csr_") && !Set("csr_data", "csr_illegal", "csr_memoryRead").contains(path))
      ))
  }

  var inputOffset  = 0
  var outputOffset = 0
  val definitions  = scala.collection.mutable.ArrayBuffer[String]()
  val ports        = scala.collection.mutable.ArrayBuffer[String]()
  val offsets      = scala.collection.mutable.Map[String, Int]()
  for ((name, width, input) <- fields(new PrivilegedIO(p), "")) {
    val direction = if (input) "IN" else "OUT"
    val signal    = if (input) "inputs" else "outputs"
    val offset    = if (input) inputOffset else outputOffset
    definitions += s"`define PRIV_${direction}_${name.toUpperCase}_OFFSET $offset"
    definitions += s"`define PRIV_${direction}_${name.toUpperCase}_WIDTH $width"
    ports += s".io_$name(bus.$signal[$offset +: $width])"
    if (input) inputOffset += width else outputOffset += width
    offsets(name) = offset
  }

  val pmpOffsets = (0 until p.permission.entries).map { index =>
    val names = Seq("cfg_l", "cfg_res", "cfg_a", "cfg_x", "cfg_w", "cfg_r", "addr", "mask")
    names.map(name => offsets(s"pmp_${index}_$name")).mkString("'{", ",", "}")
  }.mkString("'{", ",", "}")

  definitions += s"`define PRIV_INPUT_BITS $inputOffset"
  definitions += s"`define PRIV_OUTPUT_BITS $outputOffset"
  definitions += s"`define PRIV_ADDRESS_BITS ${p.permission.addressBits}"
  definitions += s"`define PRIV_PMP_OFFSETS $pmpOffsets"
  Files.writeString(Paths.get(output, "parameters.svh"), definitions.mkString("\n") + "\n")
  Files.writeString(Paths.get(output, "ports.svh"), ports.mkString(",\n") + "\n")
}
