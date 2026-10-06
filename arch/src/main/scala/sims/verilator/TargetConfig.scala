package sims.verilator

import _root_.circt.stage.ChiselStage
import sims.soc.SystemTarget

//===----------------------------------------------------------------------===//
object Elaborate extends App {
  if (args.isEmpty) {
    println("Usage: Elaborate <full.SystemTarget.ClassName> [firtool-opts...]")
    println("Example: Elaborate sims.verilator.GobanVerilatorTarget")
    sys.exit(1)
  }
  val targetClassName = args(0)
  println(s"Elaborating BBSimHarness with target: $targetClassName")

  val target: SystemTarget =
    try {
      Class.forName(targetClassName).getDeclaredConstructor().newInstance() match {
        case system: SystemTarget => system
        case _ =>
          println(s"Error: $targetClassName is not a SystemTarget")
          sys.exit(1)
      }
    } catch {
      case e: ClassNotFoundException =>
        println(s"Error: target class not found: $targetClassName")
        sys.exit(1)
    }

  val rawFirtoolOpts = args.drop(1)

  val outDir = rawFirtoolOpts.collectFirst {
    case opt if opt.startsWith("-o=") => opt.stripPrefix("-o=")
  }.getOrElse {
    throw new Exception("missing -o=<dir> in firtool opts")
  }

  val firtoolOpts = rawFirtoolOpts.filterNot { opt =>
    opt == "--split-verilog" || opt == "--difftest" || opt.startsWith("-o=")
  }

  ChiselStage.emitSystemVerilogFile(
    new SystemHarness(target, diffTest = rawFirtoolOpts.contains("--difftest")),
    firtoolOpts = firtoolOpts,
    args = Array("--target-dir", outDir, "--split-verilog")
  )
}
