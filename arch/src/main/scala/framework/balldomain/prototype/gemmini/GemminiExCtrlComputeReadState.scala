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
      io.bankReadReq(0).valid     := true.B
      io.bankReadReq(0).bits.addr := read_row_cnt
      io.bankReadReq(1).valid     := true.B
      io.bankReadReq(1).bits.addr := read_row_cnt
      when(io.bankReadReq(0).ready && io.bankReadReq(1).ready) {
        read_row_cnt := read_row_cnt + 1.U
      }
    }.otherwise {
      state := sComputeFeed
    }
  }

}
