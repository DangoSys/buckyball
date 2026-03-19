package framework.balldomain.prototype.gemmini

import chisel3._

/** Minimal tag for MeshWithDelays that satisfies TagQueueTag */
class SimpleTag extends Bundle with gemmini.TagQueueTag {
  val rob = UInt(8.W)
  override def make_this_garbage(dummy: Int = 0): Unit =
    rob := 0xff.U
}

/** Sub-command encoding within the special field */
object GemminiSubCmd {
  val CONFIG              = 0.U(4.W)
  val PRELOAD             = 1.U(4.W)
  val COMPUTE_PRELOADED   = 2.U(4.W)
  val COMPUTE_ACCUMULATED = 3.U(4.W)
  val FLUSH               = 4.U(4.W)
}
