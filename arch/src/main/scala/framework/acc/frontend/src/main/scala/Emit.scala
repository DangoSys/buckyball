package framework.acc.frontend

import sims.configs.SimParam
import _root_.configs.GlobalConfig

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
      new Frontend(
        thread.acc.get.frontend,
        thread.acc.get.memDomain,
        thread.acc.get.gpDomain,
        thread.acc.get.ballDomain,
        core.privileged.permission,
        b.sim.copy(diffTest = false)
      ),
      firtoolOpts = firtoolOpts,
      args = Array("--target-dir", directory, "--split-verilog")
    )
  }

}
