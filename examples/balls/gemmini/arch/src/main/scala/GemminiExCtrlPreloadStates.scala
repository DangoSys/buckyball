package examples.balls.gemmini

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
        // The mesh expects ordinary WS weights bottom-up. For b_transpose we
        // first capture the source matrix in natural row order and explicitly
        // feed its columns bottom-up in sPreloadFeed.
        io.bankReadReq(0).bits.addr := Mux(
          cfg_bd_transpose,
          read_row_cnt,
          total_rows - 1.U - read_row_cnt
        )
        when(io.bankReadReq(0).ready) {
          read_row_cnt := read_row_cnt + 1.U
        }
      }.otherwise {
        state := sPreloadFeed
      }
    }
  }

  protected def handlePreloadFeedState(): Unit = {
    val explicitBTranspose =
      cfg_dataflow === Dataflow.WS.id.U && cfg_bd_transpose

    when(!req_sent) {
      mesh.io.req.valid                     := true.B
      mesh.io.req.bits.pe_control.dataflow  := cfg_dataflow
      mesh.io.req.bits.pe_control.propagate := 1.U
      mesh.io.req.bits.pe_control.shift     := cfg_in_shift
      mesh.io.req.bits.a_transpose          := Mux(
        cfg_dataflow === Dataflow.OS.id.U,
        true.B,
        cfg_a_transpose
      )
      // B transpose is materialized in op2Buf below. Keeping the mesh-side
      // transposer disabled avoids the unsupported WS transpose request path.
      mesh.io.req.bits.bd_transpose         := false.B
      mesh.io.req.bits.total_rows           := total_rows
      mesh.io.req.bits.tag.rob              := robIdAsTag8(rob_id_reg)
      mesh.io.req.bits.flush                := 0.U
      when(mesh.io.req.fire) {
        req_sent := true.B
      }
    }

    when(req_sent && explicitBTranspose && !xpose_ready) {
      when(xpose_row_cnt < total_rows && rdQueue0.io.deq.valid) {
        op2Buf(xpose_row_cnt) := rdQueue0.io.deq.bits.data
          .asTypeOf(Vec(DIM, inputType))
        rdQueue0.io.deq.ready := true.B
        xpose_row_cnt         := xpose_row_cnt + 1.U
      }.elsewhen(xpose_row_cnt >= total_rows) {
        xpose_ready := true.B
      }
    }

    when(
      req_sent && (!explicitBTranspose || xpose_ready) && feed_row_cnt < total_rows
    ) {
      when(explicitBTranspose || rdQueue0.io.deq.valid) {
        val row_data        = rdQueue0.io.deq.bits.data.asTypeOf(Vec(DIM, inputType))
        val transpose_col   = total_rows - 1.U - feed_row_cnt
        val transposed_data =
          VecInit((0 until DIM).map(i => op2Buf(i)(transpose_col)))
        val weight_data     = Mux(explicitBTranspose, transposed_data, row_data)
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
          VecInit(
            weight_data.grouped(config.tileColumns).map(g => VecInit(g)).toSeq
          )
        )
        when(mesh.io.a.ready && mesh.io.b.ready && mesh.io.d.ready) {
          rdQueue0.io.deq.ready := !explicitBTranspose
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
