package framework.core.tbus

import sims.configs.SimParam
import _root_.configs.GlobalConfig
import java.nio.file.{Files, Paths}
import upickle.default.read

object Emit {

  def emit(
    b:           GlobalConfig,
    hartId:      Int,
    threadId:    Int,
    directory:   String,
    firtoolOpts: Array[String]
  ): Unit = {
    val core   =
      b.device.cores.find(_.hartId == hartId).getOrElse(throw new IllegalArgumentException(s"Unknown hart ID: $hartId"))
    val thread =
      core.threads.find(_.id == threadId).getOrElse(throw new IllegalArgumentException(s"Unknown thread ID: $threadId"))

    _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
      new SharedNetwork(
        thread.acc.get.frontend,
        thread.acc.get.memDomain,
        core.tbus.get,
        core.threads.map(_.id),
        core.threads.map(_.id),
        b.sim
      ),
      firtoolOpts = firtoolOpts,
      args = Array("--target-dir", directory, "--split-verilog")
    )
  }

}
