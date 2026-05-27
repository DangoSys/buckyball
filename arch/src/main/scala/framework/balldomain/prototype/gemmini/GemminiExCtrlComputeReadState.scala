package framework.balldomain.prototype.gemmini

import chisel3._
import chisel3.util._
import gemmini._

trait GemminiExCtrlComputeReadState { this: GemminiExCtrl =>

  protected def handleComputeReadState(): Unit = {
    // Drain stale preload data from shared queues before issuing compute reads
    when(read_row_cnt === 0.U && (rdQueue0.io.deq.valid || rdQueue1.io.deq.valid)) {
      rdQueue0.io.deq.ready := true.B
      rdQueue1.io.deq.ready := true.B
    }.elsewhen(read_row_cnt < total_rows) {
      val op1Zero = zero_op1_tail && read_row_cnt =/= 0.U
      io.bankReadReq(0).valid     := !op1Zero && !read_done(0)
      io.bankReadReq(0).bits.addr := read_row_cnt
      io.bankReadReq(1).valid     := !zero_op2 && !read_done(1)
      io.bankReadReq(1).bits.addr := read_row_cnt
      when(io.bankReadReq(0).fire) {
        read_done(0) := true.B
      }
      when(io.bankReadReq(1).fire) {
        read_done(1) := true.B
      }
      when(
        (op1Zero || read_done(0) || io.bankReadReq(0).fire) && (zero_op2 || read_done(1) || io.bankReadReq(1).fire)
      ) {
        read_row_cnt        := read_row_cnt + 1.U
        read_done.foreach(_ := false.B)
      }
    }.otherwise {
      state := sComputeFeed
    }
  }

}
