package framework.balldomain.prototype.gemmini

import chisel3._
import chisel3.util._
import gemmini._

trait GemminiExCtrlCmdStates { this: GemminiExCtrl =>

  protected def handleIdleState(): Unit = {
    io.cmdReq.ready := true.B
    when(io.cmdReq.fire) {
      rob_id_reg     := io.cmdReq.bits.rob_id
      is_sub_reg     := io.cmdReq.bits.is_sub
      sub_rob_id_reg := io.cmdReq.bits.sub_rob_id
      op1_bank       := io.cmdReq.bits.cmd.op1_bank
      op2_bank       := io.cmdReq.bits.cmd.op2_bank
      wr_bank        := io.cmdReq.bits.cmd.wr_bank
      total_rows     := Mux(io.cmdReq.bits.cmd.iter === 0.U, DIM.U, io.cmdReq.bits.cmd.iter)

      when(sub_cmd === GemminiSubCmd.CONFIG) {
        cfg_dataflow     := io.cmdReq.bits.cmd.special(4)
        cfg_a_transpose  := io.cmdReq.bits.cmd.special(7)
        cfg_bd_transpose := io.cmdReq.bits.cmd.special(8)
        cfg_in_shift     := io.cmdReq.bits.cmd.special(log2Up(config.accWidth) + 8, 9)
        io.cmdResp.valid := true.B
        state            := sCommit
      }.elsewhen(sub_cmd === GemminiSubCmd.PRELOAD) {
        read_row_cnt := 0.U
        feed_row_cnt := 0.U
        req_sent     := false.B
        state        := sPreloadRead
      }.elsewhen(sub_cmd === GemminiSubCmd.COMPUTE_PRELOADED || sub_cmd === GemminiSubCmd.COMPUTE_ACCUMULATED) {
        read_row_cnt    := 0.U
        feed_row_cnt    := 0.U
        outBufRows      := 0.U
        outBufCollected := 0.U
        req_sent        := false.B
        state           := sComputeRead
      }.elsewhen(sub_cmd === GemminiSubCmd.FLUSH) {
        state := sFlush
      }
    }
  }

  protected def handleCommitState(): Unit = {
    io.cmdResp.valid := true.B
    when(io.cmdResp.fire) {
      state := sIdle
    }
  }

  protected def handleFlushState(): Unit = {
    mesh.io.req.valid           := true.B
    mesh.io.req.bits.flush      := 2.U
    mesh.io.req.bits.total_rows := DIM.U
    mesh.io.req.bits.tag.rob    := robIdAsTag8(rob_id_reg)

    mesh.io.a.valid := true.B
    mesh.io.a.bits  := 0.U.asTypeOf(mesh.A_TYPE)
    mesh.io.b.valid := true.B
    mesh.io.b.bits  := 0.U.asTypeOf(mesh.B_TYPE)
    mesh.io.d.valid := true.B
    mesh.io.d.bits  := 0.U.asTypeOf(mesh.D_TYPE)

    when(mesh.io.req.ready) {
      io.cmdResp.valid := true.B
      when(io.cmdResp.fire) {
        state := sIdle
      }
    }
  }

}
