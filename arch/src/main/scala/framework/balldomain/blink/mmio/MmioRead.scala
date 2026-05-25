package framework.balldomain.blink.mmio

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.balldomain.blink.{HasBallId, HasRobId}
import framework.memdomain.backend.mmio.{MmioReadReq, MmioReadResp}

/**
 * MmioRead: Ball-facing MMIO read port (part of BlinkIO).
 *
 * Defined from the CONSUMER (MmioRouter) perspective:
 *   - req is Flipped(Decoupled) so the consumer drives ready and reads valid/bits
 *   - resp is Decoupled so the consumer drives valid/bits and reads ready
 *   - ball_id, rob_id, meta_bank are Input (driven by the Ball when consumed)
 *
 * BlinkIO wraps this in Flipped(...), so from the Ball's perspective:
 *   - req becomes Decoupled (Ball drives valid/bits, reads ready)
 *   - resp becomes Flipped(Decoupled) (Ball reads valid/bits, drives ready)
 *   - ball_id, rob_id, meta_bank become Output (Ball drives them)
 */
class MmioRead(val b: GlobalConfig) extends Bundle with HasBallId with HasRobId {
  val req       = Flipped(Decoupled(new MmioReadReq(b)))
  val resp      = Decoupled(new MmioReadResp(b))
  val meta_bank = Input(UInt(log2Up(b.memDomain.bankNum).W))
}

object MmioRead {

  /**
   * Tie off an MmioRead port from the Ball's perspective (BlinkIO.mmioRead).
   *  Ball's perspective after Flipped:
   *    - req: Decoupled (Ball drives) - tie valid=0, bits=0
   *    - resp: Flipped(Decoupled) (Ball reads) - tie ready=0
   *    - ball_id, rob_id, meta_bank: Output (Ball drives) - tie to 0
   */
  def tieOff(port: MmioRead): Unit = {
    port.req.valid  := false.B
    port.req.bits   := 0.U.asTypeOf(port.req.bits)
    port.resp.ready := false.B
    port.ball_id    := 0.U
    port.rob_id     := 0.U
    port.meta_bank  := 0.U
  }

}
