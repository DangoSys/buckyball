package framework.balldomain.prototype.gemmini

import chisel3._
import chisel3.util._
import gemmini._

trait GemminiExCtrlComputeFeedState { this: GemminiExCtrl =>

  protected def handleComputeFeedState(): Unit = {
    // OS: do not collect mesh rows here — preload matmul responses still drain during feed;
    // sComputeFlush drops tag=0xff (garbage) resps and collects non-garbage rows only.

    when(!req_sent) {
      mesh.io.req.valid                     := true.B
      mesh.io.req.bits.pe_control.dataflow  := cfg_dataflow
      mesh.io.req.bits.pe_control.propagate := 1.U
      mesh.io.req.bits.pe_control.shift     := cfg_in_shift
      mesh.io.req.bits.a_transpose          := Mux(cfg_dataflow === Dataflow.OS.id.U, !cfg_a_transpose, cfg_a_transpose)
      mesh.io.req.bits.bd_transpose         := cfg_bd_transpose
      mesh.io.req.bits.total_rows           := total_rows
      mesh.io.req.bits.tag.rob              := robIdAsTag8(rob_id_reg)
      mesh.io.req.bits.flush                := 0.U
      when(mesh.io.req.fire) {
        req_sent := true.B
      }
    }

    when(req_sent && feed_row_cnt < total_rows) {
      when(rdQueue0.io.deq.valid && rdQueue1.io.deq.valid) {
        val a_row = rdQueue0.io.deq.bits.data.asTypeOf(Vec(DIM, inputType))
        val x_row = rdQueue1.io.deq.bits.data.asTypeOf(Vec(DIM, inputType))
        mesh.io.a.valid := true.B
        mesh.io.a.bits  := VecInit(a_row.grouped(config.tileRows).map(g => VecInit(g)).toSeq)
        when(cfg_dataflow === Dataflow.OS.id.U) {
          // OS: stream A/B, D=0
          mesh.io.b.valid := true.B
          mesh.io.b.bits  := VecInit(x_row.grouped(config.tileColumns).map(g => VecInit(g)).toSeq)
          mesh.io.d.valid := true.B
          mesh.io.d.bits  := 0.U.asTypeOf(mesh.D_TYPE)
        }.otherwise {
          // WS: stream A/D, B comes from preloaded weights
          mesh.io.b.valid := true.B
          mesh.io.b.bits  := 0.U.asTypeOf(mesh.B_TYPE)
          mesh.io.d.valid := true.B
          mesh.io.d.bits  := VecInit(x_row.grouped(config.tileColumns).map(g => VecInit(g)).toSeq)
        }
        when(mesh.io.a.ready && mesh.io.b.ready && mesh.io.d.ready) {
          rdQueue0.io.deq.ready := true.B
          rdQueue1.io.deq.ready := true.B
          feed_row_cnt          := feed_row_cnt + 1.U
        }
      }
    }

    when(req_sent && feed_row_cnt >= total_rows) {
      when(cfg_dataflow === Dataflow.OS.id.U) {
        outBufRows   := total_rows - 1.U
        req_sent     := false.B
        feed_row_cnt := 0.U
        state        := sComputeFlush
      }.otherwise {
        outBufRows := 0.U
        state      := sDrain
      }
    }
  }

}
