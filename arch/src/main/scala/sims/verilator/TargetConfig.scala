package sims.verilator

import _root_.circt.stage.ChiselStage
import sims.soc.SystemTarget
import framework.system.configloader.ChipLoader
import java.nio.file.{Files, Paths}

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
  val mainOnly       = rawFirtoolOpts.contains("--main-only")

  val outDir = rawFirtoolOpts.collectFirst {
    case opt if opt.startsWith("-o=") => opt.stripPrefix("-o=")
  }.getOrElse {
    throw new Exception("missing -o=<dir> in firtool opts")
  }

  val firtoolOpts = rawFirtoolOpts.filterNot { opt =>
    opt == "--split-verilog" || opt == "--difftest" || opt == "--main-only" || opt.startsWith("-o=")
  }

  ChiselStage.emitSystemVerilogFile(
    new SystemHarness(target, diffTest = rawFirtoolOpts.contains("--difftest"), mainOnly = mainOnly),
    firtoolOpts = firtoolOpts,
    args = Array("--target-dir", outDir, "--split-verilog")
  )
  val hierarchy = Paths.get(outDir).resolve("hierarchy.vlt")
  if (!mainOnly && ChipLoader.load(target.pb).tiles.exists(!_.main)) {
    Files.writeString(hierarchy, "`verilator_config\nhier_block -module \"ComputeTile\"\n")
  } else {
    Files.deleteIfExists(hierarchy)
  }
}
