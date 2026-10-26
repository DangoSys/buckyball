package framework.core.execution.register_file

import framework.core.execution.register_file.configs.Params
import java.nio.file.{Files, Paths}
import upickle.default.read

object Emit extends App {
  require(args.length >= 2 && args(0) == "--config")
  val param = read[Params](Files.readString(Paths.get(args(1))))
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new RegisterFile(param),
    firtoolOpts = args.drop(2),
    args = Array("--target-dir", "build/RegisterFile", "--split-verilog")
  )
}
