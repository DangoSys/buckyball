package examples.balls.gemmini

import chisel3._
import chisel3.util._

trait GemminiExCtrlAccumulateState { this: GemminiExCtrl =>

  protected def handleReadAccumState(): Unit = {
    when(!acc_read_pending) {
      io.bankReadReq(0).valid     := true.B
      io.bankReadReq(0).bits.addr := wr_base + store_row_cnt
      when(io.bankReadReq(0).fire) {
        acc_read_pending := true.B
      }
    }.otherwise {
      rdQueue0.io.deq.ready := true.B
      when(rdQueue0.io.deq.fire) {
        val elemsPerPort = b.memDomain.bankWidth / config.accWidth
        val previous     = rdQueue0.io.deq.bits.data.asTypeOf(Vec(elemsPerPort, accType))
        for ((value, index) <- outBuf(store_row_cnt).flatten.zipWithIndex) {
          when(acc_read_group === (index / elemsPerPort).U) {
            value := value + previous(index % elemsPerPort)
          }
        }
        acc_read_pending := false.B
        when(acc_read_group === (outBW - 1).U) {
          acc_read_group := 0.U
          state          := sStore
        }.otherwise {
          acc_read_group := acc_read_group + 1.U
        }
      }
    }
  }

}
