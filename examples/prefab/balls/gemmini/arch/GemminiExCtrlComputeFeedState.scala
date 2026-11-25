package examples.balls.gemmini

import chisel3._
import chisel3.util._
import gemmini._

trait GemminiExCtrlComputeFeedState { this: GemminiExCtrl =>

  protected def handleComputeFeedState(): Unit = {
    // OS: do not collect mesh rows here — preload matmul responses still drain during feed;
    // sComputeFlush drops tag=0xff (garbage) resps and collects non-garbage rows only.

    when(!req_sent) {
      mesh.io.req.valid                     := true.B
      mesh.io.req.bits.pe_control.dataflow  := execution_dataflow
      mesh.io.req.bits.pe_control.propagate := 1.U
      mesh.io.req.bits.pe_control.shift     := execution_shift
      mesh.io.req.bits.a_transpose          := Mux(
        execution_dataflow === Dataflow.OS.id.U,
        true.B,
        cfg_a_transpose
      )
      // b_transpose applies to the operand op2Buf holds (OS) or to the weights
      // PRELOAD feeds (WS); a compute feeds the mesh no weights to transpose.
      mesh.io.req.bits.bd_transpose         := false.B
      mesh.io.req.bits.total_rows           := total_rows
      // A request's tag labels the matmul after it (WS) or the one after that
      // (OS): the preload before a compute tags the compute's result rows. The
      // rows a compute's tag labels, the next preload's, hold no result.
      mesh.io.req.bits.tag.rob              := 0xff.U
      mesh.io.req.bits.flush                := 0.U
      when(mesh.io.req.fire) {
        req_sent := true.B
      }
    }

    val op1FromBuf = execution_dataflow === Dataflow.OS.id.U && !cfg_a_transpose
    val op2FromBuf = execution_dataflow === Dataflow.OS.id.U && cfg_bd_transpose
    val needXpose  = op1FromBuf || (op2FromBuf && !zero_op2)

    when(req_sent && needXpose && !xpose_ready) {
      // zero_op1_tail reads only op1's row 0 (sComputeRead): its other rows
      // are zero, buffered without a read.
      val op1TailZero = zero_op1_tail && xpose_row_cnt =/= 0.U
      val op1Ready    = !op1FromBuf || op1TailZero || rdQueue0.io.deq.valid
      val op2Ready    = !op2FromBuf || zero_op2 || rdQueue1.io.deq.valid
      when(xpose_row_cnt < total_rows && op1Ready && op2Ready) {
        when(op1FromBuf) {
          op1Buf(xpose_row_cnt) := Mux(
            op1TailZero,
            0.U.asTypeOf(Vec(DIM, inputType)),
            rdQueue0.io.deq.bits.data.asTypeOf(Vec(DIM, inputType))
          )
          rdQueue0.io.deq.ready := !op1TailZero
        }
        when(op2FromBuf && !zero_op2) {
          op2Buf(xpose_row_cnt) := rdQueue1.io.deq.bits.data
            .asTypeOf(Vec(DIM, inputType))
          rdQueue1.io.deq.ready := true.B
        }
        xpose_row_cnt := xpose_row_cnt + 1.U
      }.elsewhen(xpose_row_cnt >= total_rows) {
        xpose_ready := true.B
      }
    }

    when(req_sent && (!needXpose || xpose_ready) && feed_row_cnt < total_rows) {
      // A buffered op1 is fed by columns: its zero rows are already in op1Buf.
      val op1Zero = zero_op1_tail && feed_row_cnt =/= 0.U && !op1FromBuf
      when(
        (op1Zero || op1FromBuf || rdQueue0.io.deq.valid) && (zero_op2 || op2FromBuf || rdQueue1.io.deq.valid)
      ) {
        val a_mem_row = rdQueue0.io.deq.bits.data.asTypeOf(Vec(DIM, inputType))
        val x_mem_row = rdQueue1.io.deq.bits.data.asTypeOf(Vec(DIM, inputType))
        val a_col_row = VecInit((0 until DIM).map(i => op1Buf(i)(feed_row_cnt)))
        val x_col_row = VecInit((0 until DIM).map(i => op2Buf(i)(feed_row_cnt)))
        val a_row     = Mux(op1FromBuf, a_col_row, a_mem_row)
        val x_row     = Mux(op2FromBuf, x_col_row, x_mem_row)
        mesh.io.a.valid := true.B
        mesh.io.a.bits  := Mux(
          op1Zero,
          0.U.asTypeOf(mesh.io.a.bits),
          VecInit(a_row.grouped(config.tileRows).map(g => VecInit(g)).toSeq)
        )
        when(execution_dataflow === Dataflow.OS.id.U) {
          // OS: stream A/B, D=0
          mesh.io.b.valid := true.B
          mesh.io.b.bits  := VecInit(
            x_row.grouped(config.tileColumns).map(g => VecInit(g)).toSeq
          )
          mesh.io.d.valid := true.B
          mesh.io.d.bits  := 0.U.asTypeOf(mesh.io.d.bits)
        }.otherwise {
          // WS: C = A W + D. The weights W are the ones PRELOAD fed through d;
          // D (op2, zeroed by zero_op2) enters each column through b as the
          // partial sum the PEs add their products to. d carries no weights.
          mesh.io.b.valid := true.B
          mesh.io.b.bits  := Mux(
            zero_op2,
            0.U.asTypeOf(mesh.io.b.bits),
            VecInit(
              x_row.grouped(config.tileColumns).map(g => VecInit(g)).toSeq
            )
          )
          mesh.io.d.valid := true.B
          mesh.io.d.bits  := 0.U.asTypeOf(mesh.io.d.bits)
        }
        when(mesh.io.a.ready && mesh.io.b.ready && mesh.io.d.ready) {
          rdQueue0.io.deq.ready := !op1Zero && !op1FromBuf
          rdQueue1.io.deq.ready := !zero_op2 && !op2FromBuf
          feed_row_cnt          := feed_row_cnt + 1.U
        }
      }
    }

    when(req_sent && feed_row_cnt >= total_rows) {
      when(execution_dataflow === Dataflow.OS.id.U) {
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
