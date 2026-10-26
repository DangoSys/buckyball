package framework.acc.balldomain

import framework.acc.balldomain.bbus.BallFactory
import framework.acc.balldomain.ball.blink.BlinkIO

import chisel3.Module
import _root_.configs.GlobalConfig
import framework.acc.balldomain.ball.blink.HasBlink

object Emit {

  def emit(
    b:           GlobalConfig,
    factories:   Map[String, BallFactory],
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
      new BallDomain(
        thread.acc.get.frontend,
        thread.acc.get.memDomain,
        thread.acc.get.gpDomain,
        thread.acc.get.ballDomain,
        core.privileged.permission,
        factories
      ),
      firtoolOpts = firtoolOpts,
      args = Array("--target-dir", directory, "--split-verilog")
    )
  }

}
