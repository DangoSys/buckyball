package sims.verilator

import _root_.circt.stage.ChiselStage
import sims.soc.SystemTarget

/** The design owns configuration decoding and concrete component types. */
object Elaborate {

  def main(args: Array[String]): Unit = {
    require(args.length >= 6 && args(0) == "--config" && args(2) == "--target-class" && args(4) == "--target-dir")
    val original = buckyball.config.Design.parseFrom(java.nio.file.Files.readAllBytes(java.nio.file.Paths.get(args(1))))
    val options  = args.drop(6)
    val design   = original.toBuilder.setSims(original.getSims.toBuilder
      .setDiffTest(original.getSims.getDiffTest || options.contains("--difftest"))).build()
    val target   = Class.forName(args(3)).getConstructor(classOf[buckyball.config.Design])
      .newInstance(design).asInstanceOf[SystemTarget]
    emit(target, args(5), options.filterNot(_ == "--difftest"))
  }

  def emit(target: SystemTarget, directory: String, firtoolOpts: Array[String]): Unit = {
    ChiselStage.emitSystemVerilogFile(
      new SystemHarness(target.b, target),
      args = Array("--target-dir", directory, "--split-verilog"),
      firtoolOpts = firtoolOpts
    )
  }

}
