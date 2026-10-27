// Copied from riscv-boom src/main/scala/v4/exu/execution-units/execution-unit.scala and modified for Buckyball.
// Copyright (c) The Regents of the University of California (Regents).
// All Rights Reserved. See LICENSE and LICENSE.SiFive in riscv-boom for license details.

package examples.core.boom.native

//------------------------------------------------------------------------------
// Execution Units
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//
// The issue window schedules micro-ops onto a specific execution pipeline
// A given execution pipeline may contain multiple functional units; one or more
// read ports, and one or more writeports.

import scala.collection.mutable.{ArrayBuffer}

import chisel3._
import chisel3.util._

import freechips.rocketchip.rocket.{BP, CSR, SFenceReq}
import freechips.rocketchip.rocket.constants.{MemoryOpConstants}
import org.chipsalliance.cde.config.{Parameters}
import freechips.rocketchip.tile
import freechips.rocketchip.util._

import boom.v4.common._
import boom.v4.ifu.FTQInfo
import boom.v4.util._

import boom.v4.exu._
import examples.core.boom.native.RoCCShim

class UniqueExeUnit(
  val hasCSR:  Boolean = false,
  val hasMul:  Boolean = false,
  val hasDiv:  Boolean = false,
  val hasIfpu: Boolean = false,
  val hasRocc: Boolean = false
)(
  implicit p:  Parameters)
    extends ExecutionUnit("Unq")
    with HasIrfReadPorts
    with HasImmrfReadPort
    with MemoryOpConstants {
  def nReaders = 2

  val io_squash_iss = IO(Output(Bool()))
  io_squash_iss := ((io_arb_irf_reqs(0).valid && !io_arb_irf_reqs(0).ready) ||
    (io_arb_irf_reqs(1).valid && !io_arb_irf_reqs(1).ready))

  when(io_squash_iss || arb_rebusied) {
    val will_replay = arb_uop.valid && !IsKilledByBranch(io_brupdate, io_kill, arb_uop.bits) && !arb_rebusied
    arb_uop.valid                  := will_replay
    arb_uop.bits                   := UpdateBrMask(io_brupdate, arb_uop.bits)
    arb_uop.bits.iw_p1_bypass_hint := false.B
    arb_uop.bits.iw_p2_bypass_hint := false.B

    rrd_uop.valid := false.B
  }

  val exe_int_req = Wire(new FuncUnitReq(xLen))
  exe_int_req.uop       := exe_uop.bits
  exe_int_req.rs1_data  := exe_rs1_data
  exe_int_req.rs2_data  := exe_rs2_data
  exe_int_req.rs3_data  := DontCare
  exe_int_req.imm_data  := exe_imm_data
  exe_int_req.pred_data := DontCare
  exe_int_req.ftq_info  := DontCare

  val io_mul_resp =
    if (hasMul) {
      fu_types += ((FC_MUL, true.B, "IMul"))
      val imul = Module(new PipelinedMulUnit(imulLatency, xLen))
      imul.io.req.valid  := exe_uop.valid && exe_uop.bits.fu_code(FC_MUL)
      imul.io.req.bits   := exe_int_req
      imul.io.brupdate   := io_brupdate
      imul.io.kill       := io_kill
      imul.io.resp.ready := true.B

      val resp = IO(Output(Valid(new ExeUnitResp(xLen))))
      resp := imul.io.resp
      Some(resp)
    } else {
      assert(!(exe_uop.valid && exe_uop.bits.fu_code(FC_MUL)))
      None
    }

  val (io_csr_resp, io_sfence) =
    if (hasCSR) {
      fu_types += ((FC_CSR, true.B, "CSR"))
      val alu = Module(new ALUUnit(dataWidth = xLen))
      alu.io.req.valid  := exe_uop.valid && exe_uop.bits.fu_code(FC_CSR)
      alu.io.req.bits   := exe_int_req
      alu.io.resp.ready := true.B
      alu.io.brupdate   := io_brupdate
      alu.io.kill       := io_kill

      val c = IO(Output(Valid(new CSRResp)))
      c.valid     := RegNext(alu.io.resp.valid && exe_uop.bits.csr_cmd =/= CSR.N)
      c.bits.uop  := RegNext(alu.io.resp.bits.uop)
      c.bits.data := RegNext(alu.io.resp.bits.data)
      c.bits.addr := RegNext(exe_imm_data)

      val s = IO(Valid(new SFenceReq))
      s.valid     := RegNext(exe_uop.valid && exe_uop.bits.is_sfence)
      s.bits.rs1  := RegNext(exe_uop.bits.pimm(0))
      s.bits.rs2  := RegNext(exe_uop.bits.pimm(1))
      s.bits.addr := RegNext(exe_rs1_data)
      s.bits.asid := RegNext(exe_rs2_data)
      s.bits.hv   := RegNext(exe_uop.bits.mem_cmd === M_HFENCEV)
      s.bits.hg   := RegNext(exe_uop.bits.mem_cmd === M_HFENCEG)

      (Some(c), Some(s))
    } else {
      assert(!(exe_uop.valid && exe_uop.bits.fu_code(FC_CSR)))
      assert(!(exe_uop.valid && exe_uop.bits.mem_cmd === M_SFENCE))
      (None, None)
    }

  val (io_rocc_resp, io_rocc_core) =
    if (hasRocc) {
      require(hasCSR)
      val rocc_core = IO(new RoCCShimCoreIO)
      val rocc_resp = IO(Decoupled(new ExeUnitResp(xLen)))
      val rocc      = Module(new RoCCShim)
      rocc.io.req.valid := exe_uop.valid && exe_uop.bits.is_rocc
      rocc.io.req.bits  := exe_int_req
      rocc.io.brupdate  := io_brupdate // We should assert on this somewhere
      rocc.io.status    := io_status
      rocc.io.exception := io_kill
      rocc_core <> rocc.io.core

      rocc_resp <> rocc.io.resp
      (Some(rocc_resp), Some(rocc_core))
    } else {
      assert(!(exe_uop.valid && exe_uop.bits.is_rocc))
      (None, None)
    }

  val (io_ifpu_resp) =
    if (hasIfpu) {
      val ifpu_ready = Wire(Bool())
      fu_types += ((FC_I2F, ifpu_ready, "IFPU"))

      val ifpu = Module(new IntToFPUnit(latency = intToFpLatency))
      ifpu.io.req.valid           := exe_uop.valid && exe_uop.bits.fu_code(FC_I2F)
      ifpu.io.req.bits            := exe_int_req
      ifpu.io.req.bits.uop.fp_rm  := exe_uop.bits.prs2(4, 2)
      ifpu.io.req.bits.uop.fp_typ := exe_uop.bits.prs2(1, 0)
      ifpu.io.fcsr_rm             := io_fcsr_rm
      ifpu.io.brupdate            := io_brupdate
      ifpu.io.kill                := io_kill

      // buffer up results since we share write-port on integer regfile.
      val queue =
        Module(new BranchKillableQueue(
          new ExeUnitResp(xLen + 1),
          entries = intToFpLatency + 6
        )) // TODO being overly conservative
      queue.io.enq <> ifpu.io.resp
      queue.io.brupdate := io_brupdate
      queue.io.flush    := io_kill
      assert(!(queue.io.enq.valid && !queue.io.enq.ready))
      ifpu_ready        := RegNext(queue.io.count < 2.U)

      val ifpu_resp = IO(Decoupled(new ExeUnitResp(xLen + 1)))
      ifpu_resp <> queue.io.deq
      (Some(ifpu_resp))
    } else {
      assert(!(exe_uop.valid && exe_uop.bits.fu_code(FC_I2F)))
      None
    }

  val (io_div_resp) =
    if (hasDiv) {
      val div_ready = Wire(Bool())
      fu_types += ((FC_DIV, div_ready, "IDiv"))

      val div = Module(new DivUnit(xLen))
      assert(!(div.io.req.valid && !div.io.req.ready))
      div.io.req.valid := exe_uop.valid && exe_uop.bits.fu_code(FC_DIV)
      div.io.req.bits  := exe_int_req
      div.io.brupdate  := io_brupdate
      div.io.kill      := io_kill

      div_ready := (div.io.req.ready &&
        !(exe_uop.valid && exe_uop.bits.fu_code(FC_DIV)) &&
        !(rrd_uop.valid && rrd_uop.bits.fu_code(FC_DIV)) &&
        !(arb_uop.valid && arb_uop.bits.fu_code(FC_DIV)))

      val div_resp = IO(Decoupled(new ExeUnitResp(xLen)))
      div_resp <> div.io.resp
      Some(div_resp)
    } else {
      assert(!(exe_uop.valid && exe_uop.bits.fu_code(FC_DIV)))
      None
    }

  io_ready_fu_types := get_ready_fu_types()

}
