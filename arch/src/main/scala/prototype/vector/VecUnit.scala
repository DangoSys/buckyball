package prototype.vector

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}
import org.chipsalliance.cde.config.Parameters

import prototype.vector._
import framework.memdomain.backend.banks.{SramReadIO, SramWriteIO}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.blink.Status

@instantiable
class VecUnit(val parameter: BallDomainParam)(implicit p: Parameters)
    extends Module
    with SerializableModule[BallDomainParam] {
  // Derived parameters - using default values for compatibility
  val InputNum   = 16 // Default value, should be derived from parameter if needed
  val inputWidth = 8  // UInt8
  val accWidth   = 32 // UInt32
  val bankWidth  = parameter.bankWidth

  @public
  val io = IO(new Bundle {
    val cmdReq  = Flipped(Decoupled(new BallRsIssue(parameter)))
    val cmdResp = Decoupled(new BallRsComplete(parameter))

    // Connect to unified bank read/write interface
    val bankRead  = Vec(parameter.numBanks, Flipped(new SramReadIO(parameter.bankEntries, bankWidth)))
    // Connect to unified bank write interface
    val bankWrite =
      Vec(parameter.numBanks, Flipped(new SramWriteIO(parameter.bankEntries, accWidth, parameter.bankMaskLen)))

    // Status output
    val status = new Status
  })

// -----------------------------------------------------------------------------
// VECCTRLUNIT
// -----------------------------------------------------------------------------
  val VecCtrlUnit: Instance[VecCtrlUnit] = Instantiate(new VecCtrlUnit(parameter))
  VecCtrlUnit.io.cmdReq <> io.cmdReq
  io.cmdResp <> VecCtrlUnit.io.cmdResp_o

// -----------------------------------------------------------------------------
// VECLOADUNIT
// -----------------------------------------------------------------------------
  val VecLoadUnit: Instance[VecLoadUnit] = Instantiate(new VecLoadUnit(parameter))
  VecLoadUnit.io.ctrl_ld_i <> VecCtrlUnit.io.ctrl_ld_o
  for (i <- 0 until parameter.numBanks) {
    io.bankRead(i).req <> VecLoadUnit.io.bankReadReq(i)
    VecLoadUnit.io.bankReadResp(i) <> io.bankRead(i).resp
  }

// -----------------------------------------------------------------------------
// VECEX
// -----------------------------------------------------------------------------
  val VecEX: Instance[VecEXUnit] = Instantiate(new VecEXUnit(parameter))
  VecEX.io.ctrl_ex_i <> VecCtrlUnit.io.ctrl_ex_o
  VecEX.io.ld_ex_i <> VecLoadUnit.io.ld_ex_o

// -----------------------------------------------------------------------------
// VECSTOREUNIT
// -----------------------------------------------------------------------------
  val VecStoreUnit: Instance[VecStoreUnit] = Instantiate(new VecStoreUnit(parameter))
  VecStoreUnit.io.ctrl_st_i <> VecCtrlUnit.io.ctrl_st_o
  VecStoreUnit.io.ex_st_i <> VecEX.io.ex_st_o
  for (i <- 0 until parameter.numBanks) {
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
