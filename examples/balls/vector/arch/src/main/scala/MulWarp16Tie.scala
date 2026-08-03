package examples.balls.vector

import chisel3._
import chisel3.util._
import framework.balldomain.blink.{BankRead, BankWrite}
import framework.top.GlobalConfig

object MulWarp16Tie {
  def offRead(ports: Vec[BankRead], robId: UInt): Unit = {
    for (p <- ports) {
      p.rob_id           := robId
      p.ball_id          := 0.U
      p.bank_id          := 0.U
      p.group_id         := 0.U
      p.io.req.valid     := false.B
      p.io.req.bits.addr := 0.U
      p.io.resp.ready    := false.B
    }
  }

  def offWrite(
      ports: Vec[BankWrite],
      robId: UInt,
      wrBank: UInt,
      b: GlobalConfig
  ): Unit = {
    for ((p, i) <- ports.zipWithIndex) {
      p.rob_id           := robId
      p.ball_id          := 0.U
      p.bank_id          := wrBank
      p.group_id         := i.U
      p.io.req.valid     := false.B
      p.io.req.bits.addr := 0.U
      p.io.req.bits.data := 0.U
      p.io.req.bits.mask := VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B))
      p.io.resp.ready    := false.B
    }
  }
}
