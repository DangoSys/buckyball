package framework.balldomain.prototype.gemmini

import chisel3._
import chisel3.util._
import gemmini._

trait GemminiExCtrlStoreOps { this: GemminiExCtrl =>

  protected def handleComputeFlushState(): Unit = {
    when(mesh.io.resp.fire && mesh.io.resp.bits.total_rows === total_rows) {
      when(outBufCollected < total_rows && mesh.io.resp.bits.tag.rob =/= 0xff.U) {
        outBuf(total_rows - 1.U - outBufCollected) := widenMeshToAcc(mesh.io.resp.bits.data)
        outBufCollected                            := outBufCollected + 1.U
      }.otherwise {}
    }

    when(outBufCollected >= total_rows) {
      store_row_cnt          := 0.U
      port_written.foreach(_ := false.B)
      state                  := sStore
    }.otherwise {
      // Flush once to drain remaining rows after feeding is complete.
      when(!req_sent) {
        mesh.io.req.valid                     := true.B
        mesh.io.req.bits.pe_control.dataflow  := cfg_dataflow
        mesh.io.req.bits.pe_control.propagate := 1.U
        mesh.io.req.bits.pe_control.shift     := cfg_in_shift
        mesh.io.req.bits.a_transpose          := Mux(cfg_dataflow === Dataflow.OS.id.U, !cfg_a_transpose, cfg_a_transpose)
        mesh.io.req.bits.bd_transpose         := cfg_bd_transpose
        mesh.io.req.bits.total_rows           := total_rows
        mesh.io.req.bits.tag.rob              := robIdAsTag8(rob_id_reg)
        mesh.io.req.bits.flush                := 1.U
        when(mesh.io.req.fire) {
          req_sent := true.B
        }
      }
    }
  }

  protected def handleDrainState(): Unit = {
    when(mesh.io.resp.valid) {
      when(cfg_dataflow === Dataflow.OS.id.U) {
        when(mesh.io.resp.bits.total_rows === total_rows) {
          outBuf(outBufRows) := widenMeshToAcc(mesh.io.resp.bits.data)
          outBufRows         := outBufRows - 1.U
          outBufCollected    := outBufCollected + 1.U
        }
      }.otherwise {
        outBuf(outBufRows) := widenMeshToAcc(mesh.io.resp.bits.data)
        outBufRows         := outBufRows + 1.U
      }
    }

    when(cfg_dataflow === Dataflow.OS.id.U) {
      when(outBufCollected >= total_rows) {
        store_row_cnt          := 0.U
        port_written.foreach(_ := false.B)
        state                  := sStore
      }
    }.otherwise {
      when(outBufRows >= total_rows) {
        store_row_cnt          := 0.U
        port_written.foreach(_ := false.B)
        state                  := sStore
      }
    }
  }

  protected def handleStoreState(): Unit = {
    when(store_row_cnt < total_rows) {
      // One row = DIM elements; split across outBW ports (group_id 0..outBW-1).
      // mvout reads (addr, group 0), (addr, group 1), ... and concatenates as row.
      // So port i (group_id=i) must get elements [i*elemsPerPort .. (i+1)*elemsPerPort).
      val rowIdx   = store_row_cnt
      val row      = outBuf(rowIdx)
      val flat_raw = VecInit(row.flatten)
      val row_bits = Cat(flat_raw.map(_.asUInt).reverse)

      val bitsPerPort = b.memDomain.bankWidth

      for (i <- 0 until outBW) {
        when(!port_written(i)) {
          val slice = row_bits((i + 1) * bitsPerPort - 1, i * bitsPerPort)
          io.bankWrite(i).req.valid      := true.B
          io.bankWrite(i).req.bits.addr  := store_row_cnt
          io.bankWrite(i).req.bits.data  := slice
          io.bankWrite(i).req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(true.B))
          io.bankWrite(i).req.bits.wmode := true.B
          when(io.bankWrite(i).req.ready) {
            port_written(i) := true.B
          }
        }
      }

      when(port_written.asUInt.andR) {
        store_row_cnt          := store_row_cnt + 1.U
        port_written.foreach(_ := false.B)
      }
    }.otherwise {
      state := sCommit
    }
  }

}
