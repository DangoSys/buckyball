package sims.p2e

import _root_.circt.stage.ChiselStage

//===----------------------------------------------------------------------===//
object Elaborate extends App {
  if (args.isEmpty) {
    println("Usage: Elaborate <full.P2ETarget.ClassName> [--difftest] [firtool-opts...] -o=<dir>")
    println("Example: Elaborate sims.p2e.PebbleP2ETarget")
    sys.exit(1)
  }
  val targetClassName = args(0)
  println(s"Elaborating P2ETop with target: $targetClassName")

  val target: P2ETarget =
    try {
      Class.forName(targetClassName).getDeclaredConstructor().newInstance() match {
        case p2e: P2ETarget => p2e
        case _ =>
          println(s"Error: $targetClassName is not a P2ETarget")
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
    new P2ETop(target, diffTest = rawFirtoolOpts.contains("--difftest")),
    firtoolOpts = firtoolOpts,
    args = Array("--target-dir", outDir, "--split-verilog")
  )
}
