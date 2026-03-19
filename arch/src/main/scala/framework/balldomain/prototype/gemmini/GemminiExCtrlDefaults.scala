package framework.balldomain.prototype.gemmini

import chisel3._
import chisel3.util._

trait GemminiExCtrlDefaults { this: GemminiExCtrl =>

  protected def applyDefaults(): Unit = {
    io.cmdReq.ready := state === sIdle

    io.cmdResp.valid           := false.B
    io.cmdResp.bits.rob_id     := rob_id_reg
    io.cmdResp.bits.is_sub     := is_sub_reg
    io.cmdResp.bits.sub_rob_id := sub_rob_id_reg

    for (i <- 0 until inBW) {
      io.bankReadReq(i).valid     := false.B
      io.bankReadReq(i).bits.addr := 0.U
    }

    rdQueue0.io.deq.ready := false.B
    rdQueue1.io.deq.ready := false.B

    mesh.io.a.valid   := false.B
    mesh.io.a.bits    := 0.U.asTypeOf(mesh.A_TYPE)
    mesh.io.b.valid   := false.B
    mesh.io.b.bits    := 0.U.asTypeOf(mesh.B_TYPE)
    mesh.io.d.valid   := false.B
    mesh.io.d.bits    := 0.U.asTypeOf(mesh.D_TYPE)
    mesh.io.req.valid := false.B
    mesh.io.req.bits  := 0.U.asTypeOf(mesh.io.req.bits)
    // mesh.io.resp is Valid-only (no ready); MeshWithDelays assumes immediate observation when valid

    io.bankWrite.foreach { bw =>
      bw.req.valid      := false.B
      bw.req.bits.addr  := 0.U
      bw.req.bits.data  := 0.U
      bw.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B))
      bw.req.bits.wmode := true.B
      bw.resp.ready     := true.B
    }
  }

}
