package framework.balldomain.prototype.vector

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

import framework.balldomain.prototype.vector._
import framework.memdomain.backend.banks.{SramReadIO, SramWriteIO}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.top.GlobalConfig
import framework.balldomain.blink.Status
import framework.balldomain.prototype.vector.configs.VectorBallParam

@instantiable
class VecUnit(val b: GlobalConfig) extends Module {
  // Get parameters from ball's own config JSON file
  val ballConfig = VectorBallParam()
  val InputNum   = ballConfig.lane
  val inputWidth = ballConfig.inputWidth
  val accWidth   = ballConfig.outputWidth
  val bankWidth  = b.memDomain.bankWidth

  @public
  val io = IO(new Bundle {
    val cmdReq  = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp = Decoupled(new BallRsComplete(b))

    // Connect to unified bank read/write interface
    val bankRead  = Vec(b.memDomain.bankNum, Flipped(new SramReadIO(b)))
    // Connect to unified bank write interface
    val bankWrite =
      Vec(b.memDomain.bankNum, Flipped(new SramWriteIO(b)))

    // Status output
    val status = new Status
  })

// -----------------------------------------------------------------------------
// VECCTRLUNIT
// -----------------------------------------------------------------------------
  val VecCtrlUnit: Instance[VecCtrlUnit] = Instantiate(new VecCtrlUnit(b))
  VecCtrlUnit.io.cmdReq <> io.cmdReq
  io.cmdResp <> VecCtrlUnit.io.cmdResp_o

// -----------------------------------------------------------------------------
// VECLOADUNIT
// -----------------------------------------------------------------------------
  val VecLoadUnit: Instance[VecLoadUnit] = Instantiate(new VecLoadUnit(b))
  VecLoadUnit.io.ctrl_ld_i <> VecCtrlUnit.io.ctrl_ld_o
  for (i <- 0 until b.memDomain.bankNum) {
    io.bankRead(i).req <> VecLoadUnit.io.bankReadReq(i)
    VecLoadUnit.io.bankReadResp(i) <> io.bankRead(i).resp
  }

// -----------------------------------------------------------------------------
// VECEX
// -----------------------------------------------------------------------------
  val VecEX: Instance[VecEXUnit] = Instantiate(new VecEXUnit(b))
  VecEX.io.ctrl_ex_i <> VecCtrlUnit.io.ctrl_ex_o
  VecEX.io.ld_ex_i <> VecLoadUnit.io.ld_ex_o

// -----------------------------------------------------------------------------
// VECSTOREUNIT
// -----------------------------------------------------------------------------
  val VecStoreUnit: Instance[VecStoreUnit] = Instantiate(new VecStoreUnit(b))
  VecStoreUnit.io.ctrl_st_i <> VecCtrlUnit.io.ctrl_st_o
  VecStoreUnit.io.ex_st_i <> VecEX.io.ex_st_o
  for (i <- 0 until b.memDomain.bankNum) {
    io.bankWrite(i) <> VecStoreUnit.io.bankWrite(i)
  }
  VecCtrlUnit.io.cmdResp_i <> VecStoreUnit.io.cmdResp_o

// -----------------------------------------------------------------------------
// Set DontCare
// -----------------------------------------------------------------------------
  // for (i <- 0 until b.sp_banks) {
  //   io.sramWrite(i) := DontCare
  // }
  // for (i <- 0 until b.acc_banks) {
  //   io.accRead(i) := DontCare
  // }

// -----------------------------------------------------------------------------
// Status tracking
// -----------------------------------------------------------------------------
  val iterCnt   = RegInit(0.U(32.W))
  val hasInput  = RegInit(false.B)
  val hasOutput = RegInit(false.B)

  when(io.cmdReq.fire) {
    hasInput := true.B
  }
  when(io.cmdResp.fire) {
    hasOutput := false.B
    hasInput  := false.B
    iterCnt   := iterCnt + 1.U
  }
  when(io.cmdResp.valid && !hasOutput) {
    hasOutput := true.B
  }

  io.status.ready    := io.cmdReq.ready
  io.status.valid    := io.cmdResp.valid
  io.status.idle     := !hasInput && !hasOutput
  io.status.init     := hasInput && !hasOutput
  io.status.running  := hasOutput
  io.status.complete := io.cmdResp.fire
  io.status.iter     := iterCnt
}
