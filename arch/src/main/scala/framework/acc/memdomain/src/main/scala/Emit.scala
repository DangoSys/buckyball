package framework.acc.memdomain

import sims.configs.SimParam
import framework.core.tbus.configs.Layout

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
      new MemDomain(
        thread.acc.get.frontend,
        thread.acc.get.memDomain,
        thread.acc.get.gpDomain,
        thread.acc.get.transport,
        core.privileged.permission,
        core.memory.dcache.tracking,
        core.memory.memoryAccess.walker,
        b.sim.copy(diffTest = false)
      ),
      firtoolOpts = firtoolOpts,
      args = Array("--target-dir", directory, "--split-verilog")
    )
  }

}
