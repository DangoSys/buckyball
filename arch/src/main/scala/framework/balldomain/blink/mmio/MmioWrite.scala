package framework.balldomain.blink.mmio

import chisel3._
import chisel3.util.Decoupled
import framework.balldomain.blink.{HasBallId, HasRobId}
import framework.memdomain.backend.mmio.MmioWriteReq
import framework.top.GlobalConfig

/**
 * Ball-facing byte write port for the unified MMIO address space.
 *
 * BlinkIO wraps this in Flipped(...), so a Ball produces req and observes
 * ready. A multi-byte value uses one configured line per byte.
 */
class MmioWrite(val b: GlobalConfig) extends Bundle with HasBallId with HasRobId {
  val req = Flipped(Decoupled(new MmioWriteReq(b)))
}

object MmioWrite {

  def tieOff(port: MmioWrite): Unit = {
    port.req.valid := false.B
    port.req.bits  := 0.U.asTypeOf(port.req.bits)
    port.ball_id   := 0.U
    port.rob_id    := 0.U
  }

  def tieOff(ports: Vec[MmioWrite]): Unit = {
    for (port <- ports) {
      tieOff(port)
    }
  }

}
