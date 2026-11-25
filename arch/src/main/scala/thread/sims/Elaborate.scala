package framework.thread.sims

import _root_.circt.stage.ChiselStage
import chisel3._
import buckyball.config.{HartParamConfig, HartPlacement, SharedMemConfig}
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.contract.sims.configs.SimParam
import framework.thread.configs.ThreadParam
import framework.thread.tbus.ThreadFactory
import framework.thread.tlink.TLinkIO

/** A thread alone: its TLink is the boundary, nothing of a hart behind it. */
class ThreadAlone(
  threadp:    ThreadParam,
  permission: PmpParams,
  simp:       SimParam,
  factory:    ThreadFactory)
    extends Module {
  val tlink = IO(new TLinkIO(threadp.link, permission))
  tlink <> factory.instantiate(threadp, permission, 0, simp, 0)
}

/**
 * A thread alone, in the host of design.pb's `thread`: the placement a hart would give its one thread, without shared
 * banks. The thread's factory is its RTL class, so the entry needs no table of the threads there are.
 */
object Elaborate {

  def placement(h: buckyball.config.ThreadHost): HartPlacement =
    HartPlacement
      .newBuilder()
      .setParam(HartParamConfig.newBuilder().setXLen(h.getXLen))
      .setSharedMem(SharedMemConfig.newBuilder().setEnable(false))
      .setVirtualBankCount(h.getVirtualBankCount)
      .addThreadIndices(0)
      .setTss(h.getTss)
      .build()

  def main(args: Array[String]): Unit = {
    require(args.length >= 6 && args(0) == "--config" && args(2) == "--target-class" && args(4) == "--target-dir")
    val design     = buckyball.config.Design.parseFrom(java.nio.file.Files.readAllBytes(java.nio.file.Paths.get(args(1))))
    require(design.hasThread && design.getThreadsCount == 1, s"${args(1)}: not the design.pb of a thread alone")
    val host       = design.getThread.getHost
    val threadp    = ThreadParam.fromProto(design.getThreads(0), placement(host))
    val factory    = Class.forName(threadp.rtlClass + "$").getField("MODULE$").get(null).asInstanceOf[ThreadFactory]
    val permission = PmpParams(host.getPmpAddressBits, host.getPmpEntries)
    ChiselStage.emitSystemVerilogFile(
      new ThreadAlone(threadp, permission, SimParam.fromProto(design.getSims), factory),
      args = Array("--target-dir", args(5), "--split-verilog"),
      firtoolOpts = args.drop(6)
    )
  }

}
