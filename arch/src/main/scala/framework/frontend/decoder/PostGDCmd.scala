package framework.frontend.decoder

import chisel3._
import framework.top.GlobalConfig
import framework.frontend.scoreboard.BankAccessInfo
import framework.system.core.rocket.RoCCCommandBB

class PostGDCmd(val b: GlobalConfig) extends Bundle {
  val domain_id  = UInt(4.W)
  val ball_bid   = UInt(5.W)
  val cmd        = new RoCCCommandBB(b.tile.xLen)
  val bankAccess = new BankAccessInfo(b.frontend.bank_id_len)
  val op1_col    = UInt(b.memDomain.groupCountWidth.W)
  val op2_col    = UInt(b.memDomain.groupCountWidth.W)
  val wr_col     = UInt(b.memDomain.groupCountWidth.W)
  val isFence    = Bool()
  val isBarrier  = Bool()
}
