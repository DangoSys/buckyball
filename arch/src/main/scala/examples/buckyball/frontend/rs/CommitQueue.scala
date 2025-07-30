package buckyball.frontend.rs

import chisel3._
import chisel3.util._
import buckyball.BuckyBallConfig
import buckyball.util.Util._


class CommitQueue(queue_entries: Int = 4)(implicit bbconfig: BuckyBallConfig) extends Module {
  val rob_id_width = log2Up(bbconfig.rob_entries)
  val cmd_t = new BuckyBallCmd
  
  val io = IO(new Bundle {
    val controller_commit_i = new RSCMTInterface(rob_id_width)
    val complete_o = Valid(UInt(rob_id_width.W)) // 输出完成的rob_id
  })

  // 使用轮询仲裁器处理多个complete信号
  val arbiter = Module(new RRArbiter(new ReservationStationComplete(rob_id_width), 3))
  arbiter.io.in(0) <> io.controller_commit_i.ld
  arbiter.io.in(1) <> io.controller_commit_i.st
  arbiter.io.in(2) <> io.controller_commit_i.ex
  
  io.complete_o.valid := arbiter.io.out.valid
  io.complete_o.bits := arbiter.io.out.bits.rob_id

  arbiter.io.out.ready := true.B
}