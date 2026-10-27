package examples.balls.gemmini

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.util._
import gemmini._

/** Sub-command encoding within the special field */
object GemminiSubCmd {
  val CONFIG              = 0.U(4.W)
  val PRELOAD             = 1.U(4.W)
  val COMPUTE_PRELOADED   = 2.U(4.W)
  val COMPUTE_ACCUMULATED = 3.U(4.W)
  val FLUSH               = 4.U(4.W)
}

trait GemminiExCtrlCmdStates { this: GemminiExCtrl =>

  protected def handleIdleState(): Unit = {
    io.cmdReq.ready := true.B
    when(io.cmdReq.fire) {
      rob_id_reg       := io.cmdReq.bits.rob_id
      op1_bank         := io.cmdReq.bits.op1_bank
      op2_bank         := io.cmdReq.bits.op2_bank
      wr_bank          := io.cmdReq.bits.wr_bank
      wr_read_bank     := io.cmdReq.bits.wr_read_bank
      // rs2[41:36] and rs2[47:42] select a group of op1 and op2 within their banks.
      op1_group        := io.cmdReq.bits.op1_group + io.cmdReq.bits.rs2(41, 36)
      op2_group        := io.cmdReq.bits.op2_group + io.cmdReq.bits.rs2(47, 42)
      wr_group         := io.cmdReq.bits.wr_group
      op1_base         := io.cmdReq.bits.rs2(15, 6)
      op2_base         := io.cmdReq.bits.rs2(25, 16)
      wr_base          := io.cmdReq.bits.rs2(35, 26)
      total_rows       := Mux(
        io.cmdReq.bits.iter === 0.U,
        DIM.U,
        io.cmdReq.bits.iter
      )
      zero_op2         := io.cmdReq.bits.rs2(4)
      zero_op1_tail    := io.cmdReq.bits.rs2(5)
      accumulated_cmd  := sub_cmd === GemminiSubCmd.COMPUTE_ACCUMULATED
      reset_acc_mesh   := sub_cmd === GemminiSubCmd.COMPUTE_ACCUMULATED
      acc_read_group   := 0.U
      acc_read_pending := false.B

      when(sub_cmd === GemminiSubCmd.CONFIG) {
        cfg_dataflow     := io.cmdReq.bits.rs2(4)
        cfg_a_transpose  := io.cmdReq.bits.rs2(7)
        cfg_bd_transpose := io.cmdReq.bits.rs2(8)
        cfg_in_shift     := io.cmdReq.bits.rs2(
          log2Up(config.accWidth) + 8,
          9
        )
        state            := sCommit
      }.elsewhen(sub_cmd === GemminiSubCmd.PRELOAD) {
        assert(io.cmdReq.bits.rs2(41, 36) < io.cmdReq.bits.op1_col, "Gemmini preload op1 group outside the bank")
        read_row_cnt  := 0.U
        feed_row_cnt  := 0.U
        req_sent      := false.B
        xpose_ready   := false.B
        xpose_row_cnt := 0.U
        state         := sPreloadRead
      }.elsewhen(
        sub_cmd === GemminiSubCmd.COMPUTE_PRELOADED || sub_cmd === GemminiSubCmd.COMPUTE_ACCUMULATED
      ) {
        // A zeroed op2 (rs2[4]) reads no bank, so its group is not checked.
        assert(
          io.cmdReq.bits.rs2(41, 36) < io.cmdReq.bits.op1_col &&
            (io.cmdReq.bits.rs2(4) || io.cmdReq.bits.rs2(47, 42) < io.cmdReq.bits.op2_col),
          "Gemmini compute operand group outside the bank"
        )
        read_row_cnt        := 0.U
        feed_row_cnt        := 0.U
        outBufRows          := 0.U
        outBufCollected     := 0.U
        req_sent            := false.B
        xpose_ready         := false.B
        xpose_row_cnt       := 0.U
        read_done.foreach(_ := false.B)
        state               := Mux(sub_cmd === GemminiSubCmd.COMPUTE_ACCUMULATED, sPreloadFeed, sComputeRead)
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
    mesh.io.a.bits  := 0.U.asTypeOf(mesh.io.a.bits)
    mesh.io.b.valid := true.B
    mesh.io.b.bits  := 0.U.asTypeOf(mesh.io.b.bits)
    mesh.io.d.valid := true.B
    mesh.io.d.bits  := 0.U.asTypeOf(mesh.io.d.bits)

    when(mesh.io.req.ready) {
      io.cmdResp.valid := true.B
      when(io.cmdResp.fire) {
        state := sIdle
      }
    }
  }

}
