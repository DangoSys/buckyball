package framework.tlp.memory

import buckyball.config.Design
import chisel3._
import chisel3.reflect.DataMirror
import framework.hart.configs.HartParam
import java.nio.file.{Files, Paths}
import scala.jdk.CollectionConverters._

object Emit extends App {
  val Array("--design-pb", proto, "--output-dir", output) = args
  val design                                              = Design.parseFrom(Files.readAllBytes(Paths.get(proto)))
  val hart                                                = design.getDevice.getHartsList.asScala.find(_.getThreadIndicesCount > 0).get
  val p                                                   = HartParam.fromProto(design, hart, externalHome = hart.hasNodeId && hart.getNodeId > 0)
  require(p.threads.size >= 2, "The concurrent DMA profile requires two clients")
  val names                                               = scala.collection.mutable.ArrayBuffer[String]()
  val connections                                         = scala.collection.mutable.ArrayBuffer[String]()
  var inWidth                                             = 0
  var outWidth                                            = 0

  def bind(data: Data, path: String): Unit = data match {
    case r: Record => r.elements.foreach { case (name, field) => bind(field, s"${path}_$name") }
    case v: Vec[_] => v.toSeq.zipWithIndex.foreach { case (field, i) => bind(field, s"${path}_$i") }
    case f if f.getWidth > 0 =>
      val incoming = DataMirror.directionOf(f) == ActualDirection.Input
      val reg      = if (incoming) "inputs" else "outputs"
      val offset   = if (incoming) inWidth else outWidth
      val bits     = s"$reg[$offset +: ${f.getWidth}]"
      names += s"`define HD_${path.toUpperCase} $bits"
      connections += s".$path($bits)"
      if (incoming) inWidth += f.getWidth else outWidth += f.getWidth
    case _                   =>
  }

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    {
      val dut = new DmaMemory(
        p.interconnect.bus,
        p.privileged.permission,
        p.memory,
        p.hlink.line.addressBits,
        p.hlink.backing,
        p.threads.size
      )
      bind(dut.io, "io")
      dut
    },
    args = Array("--target-dir", s"$output/DmaMemory", "--split-verilog")
  )
  names += s"`define HD_DDR_BASE 64'h${java.lang.Long.toHexString(design.getDevice.getDdr.getBase)}"
  names += s"`define HD_INPUT_BITS $inWidth"
  names += s"`define HD_OUTPUT_BITS $outWidth"
  Files.writeString(Paths.get(output, "parameters.svh"), names.mkString("\n") + "\n")
  Files.writeString(Paths.get(output, "ports.svh"), connections.mkString(",\n") + "\n")
}
