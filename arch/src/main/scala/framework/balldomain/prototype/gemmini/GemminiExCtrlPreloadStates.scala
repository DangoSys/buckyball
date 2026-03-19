package framework.balldomain.prototype.gemmini

import chisel3._
import chisel3.util._
import gemmini._

trait GemminiExCtrlPreloadStates { this: GemminiExCtrl =>

  protected def handlePreloadReadState(): Unit = {
    when(cfg_dataflow === Dataflow.OS.id.U) {
      when(read_row_cnt < total_rows) {
        io.bankReadReq(0).valid     := true.B
        io.bankReadReq(0).bits.addr := read_row_cnt
        when(io.bankReadReq(0).ready) {
          read_row_cnt := read_row_cnt + 1.U
        }
      }.otherwise {
        state := sPreloadFeed
      }
    }.otherwise {
      when(read_row_cnt < total_rows) {
        io.bankReadReq(0).valid     := true.B
        io.bankReadReq(0).bits.addr := total_rows - 1.U - read_row_cnt
        when(io.bankReadReq(0).ready) {
          read_row_cnt := read_row_cnt + 1.U
        }
      }.otherwise {
        state := sPreloadFeed
      }
    }
  }

  protected def handlePreloadFeedState(): Unit = {
    when(!req_sent) {
      mesh.io.req.valid                     := true.B
      mesh.io.req.bits.pe_control.dataflow  := cfg_dataflow
      mesh.io.req.bits.pe_control.propagate := 1.U
      mesh.io.req.bits.pe_control.shift     := cfg_in_shift
      // Buckyball sends preload/compute as separate commands, so the
      // AlwaysOutTransposer inside MeshWithDelays cannot be primed with
      // compute-A during preload (unlike Chipyard's interleaved flow).
      // Negate here so MeshWithDelays' internal !a_transpose yields false,
      // keeping the transposer inactive.  Software must pre-transpose A.
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
      when(rdQueue0.io.deq.valid) {
        val row_data = rdQueue0.io.deq.bits.data.asTypeOf(Vec(DIM, inputType))
        mesh.io.a.valid := true.B
        mesh.io.a.bits  := 0.U.asTypeOf(mesh.A_TYPE)
        mesh.io.b.valid := true.B
        mesh.io.b.bits  := 0.U.asTypeOf(mesh.B_TYPE)
        mesh.io.d.valid := true.B
        // OS preload in Buckyball is used to prime pipeline state before compute.
        // Feed D=0 to avoid injecting bias-like data into the following matmul.
        mesh.io.d.bits  := Mux(
          cfg_dataflow === Dataflow.OS.id.U,
          0.U.asTypeOf(mesh.D_TYPE),
          VecInit(row_data.grouped(config.tileColumns).map(g => VecInit(g)).toSeq)
        )
        when(mesh.io.a.ready && mesh.io.b.ready && mesh.io.d.ready) {
          rdQueue0.io.deq.ready := true.B
          feed_row_cnt          := feed_row_cnt + 1.U
        }
      }
    }

    when(req_sent && feed_row_cnt >= total_rows) {
      io.cmdResp.valid := true.B
      when(io.cmdResp.fire) {
        state := sIdle
      }
    }
  }

}
