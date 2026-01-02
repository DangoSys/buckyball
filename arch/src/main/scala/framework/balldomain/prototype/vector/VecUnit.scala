package framework.balldomain.prototype.vector

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

import framework.balldomain.prototype.vector._
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.top.GlobalConfig
import framework.balldomain.blink.{BankRead, BankWrite, Status}
import framework.balldomain.prototype.vector.configs.VectorBallParam

@instantiable
class VecUnit(val b: GlobalConfig) extends Module {
  val ballConfig = VectorBallParam()
  val InputNum   = ballConfig.lane
  val inputWidth = ballConfig.inputWidth
  val accWidth   = ballConfig.outputWidth
  val bankWidth  = b.memDomain.bankWidth

  // Get bandwidth from config (use first VecBall mapping)
  val ballMapping = b.ballDomain.ballIdMappings.find(_.ballName == "VecBall")
    .getOrElse(throw new IllegalArgumentException("VecBall not found in config"))
  val inBW        = ballMapping.inBW
  val outBW       = ballMapping.outBW

  @public
  val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp   = Decoupled(new BallRsComplete(b))
    val bankRead  = Vec(inBW, Flipped(new BankRead(b)))
    val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))
    val status    = new Status
  })

  // Register to store rob_id when command is received
  val rob_id_reg = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  when(io.cmdReq.fire) {
    rob_id_reg := io.cmdReq.bits.rob_id
  }

  // Set rob_id for all bankRead and bankWrite channels from register
  for (i <- 0 until inBW) {
    io.bankRead(i).rob_id := rob_id_reg
  }
  for (i <- 0 until outBW) {
    io.bankWrite(i).rob_id := rob_id_reg
  }

  val VecCtrlUnit:  Instance[VecCtrlUnit]  = Instantiate(new VecCtrlUnit(b))
  val VecLoadUnit:  Instance[VecLoadUnit]  = Instantiate(new VecLoadUnit(b))
  val VecEX:        Instance[VecEXUnit]    = Instantiate(new VecEXUnit(b))
  val VecStoreUnit: Instance[VecStoreUnit] = Instantiate(new VecStoreUnit(b))

// -----------------------------------------------------------------------------
// VECCTRLUNIT
// -----------------------------------------------------------------------------

  VecCtrlUnit.io.cmdReq <> io.cmdReq
  io.cmdResp <> VecCtrlUnit.io.cmdResp_o

// -----------------------------------------------------------------------------
// VECLOADUNIT
// -----------------------------------------------------------------------------

  VecLoadUnit.io.ctrl_ld_i <> VecCtrlUnit.io.ctrl_ld_o
  for (i <- 0 until inBW) {
    io.bankRead(i).io.req <> VecLoadUnit.io.bankReadReq(i)
    VecLoadUnit.io.bankReadResp(i) <> io.bankRead(i).io.resp
    if (i == 0) {
      io.bankRead(i).bank_id := VecLoadUnit.io.op1_bank_o
    } else if (i == 1) {
      io.bankRead(i).bank_id := VecLoadUnit.io.op2_bank_o
    }
  }

// -----------------------------------------------------------------------------
// VECEX
// -----------------------------------------------------------------------------

  VecEX.io.ctrl_ex_i <> VecCtrlUnit.io.ctrl_ex_o
  VecEX.io.ld_ex_i <> VecLoadUnit.io.ld_ex_o

// -----------------------------------------------------------------------------
// VECSTOREUNIT
// -----------------------------------------------------------------------------
  VecStoreUnit.io.ctrl_st_i <> VecCtrlUnit.io.ctrl_st_o
  VecStoreUnit.io.ex_st_i <> VecEX.io.ex_st_o
  for (i <- 0 until outBW) {
    // VecUnit receives write requests from VecBall, forwards to VecStoreUnit
    // io.bankWrite is Flipped(new BankWrite(b)) so io is output from VecUnit
    // VecStoreUnit.io.bankWrite is Flipped(SramWriteIO) so it's input to VecStoreUnit
    // Connect: VecUnit outputs to VecStoreUnit inputs
    io.bankWrite(i).io <> VecStoreUnit.io.bankWrite(i)
    // Set bank_id from VecStoreUnit
    io.bankWrite(i).bank_id := VecStoreUnit.io.wr_bank_o
  }
  VecCtrlUnit.io.cmdResp_i <> VecStoreUnit.io.cmdResp_o

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
