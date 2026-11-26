package framework.device.sims.verilator

import _root_.circt.stage.ChiselStage
import framework.device.sims.soc.SystemTarget

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
      args = Array("--target-dir", directory, "--split-verilog", "--firtool-binary-path", "firtool"),
      firtoolOpts = firtoolOpts
    )
    if (target.hierBlocks.nonEmpty) {
      // A block is Verilated on one thread, and Verilator then records the cost of its updates as zero: the
      // scheduler of the top would pack many blocks into one task and evaluate them one after another. Each
      // emitted module a block pattern names gets the cost of its updates declared, in the instructions the
      // scheduler counts the logic around the blocks in: a cost far above that makes the surrounding logic
      // look free, and it ends up in one task on one thread.
      val modules = new java.io.File(directory).list().filter(_.endsWith(".sv")).map(_.stripSuffix(".sv")).sorted
      val blocks  = target.hierBlocks.flatMap { pattern =>
        val regex = pattern.replace("*", ".*")
        modules.filter(_.matches(regex))
      }.distinct
      require(blocks.nonEmpty, s"no emitted module matches the hierarchical blocks ${target.hierBlocks.mkString(", ")}")
      val costs   = blocks.flatMap { m =>
        Seq(
          s"""profile_data -hier-dpi "${m}_protectlib_seq_update" -cost 64'd1000000""",
          s"""profile_data -hier-dpi "${m}_protectlib_combo_update" -cost 64'd100000"""
        )
      }
      java.nio.file.Files.writeString(
        java.nio.file.Paths.get(directory, "hierarchy.vlt"),
        (("`verilator_config" +: target.hierBlocks.map(m => s"""hier_block -module "$m"""")) ++ costs).mkString(
          "",
          "\n",
          "\n"
        )
      )
    }
  }

}
