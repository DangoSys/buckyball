package prototype.matrix

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}
import org.chipsalliance.cde.config.Parameters

import prototype.matrix._
import framework.memdomain.backend.banks.{SramReadIO, SramWriteIO}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.blink.Status

@instantiable
class BBFP_Control(val parameter: BallDomainParam)(implicit p: Parameters)
    extends Module
    with SerializableModule[BallDomainParam] {
  // Derived parameters - using default values for compatibility
  val InputNum   = 16 // Default value
  val inputWidth = 8  // UInt8
  val accWidth   = 32 // UInt32
  val bankWidth  = parameter.bankWidth

  @public
  val io = IO(new Bundle {
    val cmdReq       = Flipped(Decoupled(new BallRsIssue(parameter)))
    val cmdResp      = Decoupled(new BallRsComplete(parameter))
    val is_matmul_ws = Input(Bool())
    // Connect to Scratchpad SRAM read/write interface
    val sramRead     = Vec(parameter.numBanks, Flipped(new SramReadIO(parameter.bankEntries, bankWidth)))
    val sramWrite    =
      Vec(parameter.numBanks, Flipped(new SramWriteIO(parameter.bankEntries, bankWidth, parameter.bankMaskLen)))

    // Connect to Accumulator write interface (unified bank now)
    val accWrite =
      Vec(parameter.numBanks, Flipped(new SramWriteIO(parameter.bankEntries, accWidth, parameter.bankMaskLen)))

    // Status output
    val status = new Status
  })

// -----------------------------------------------------------------------------
// BBFP_ID
// -----------------------------------------------------------------------------
  val BBFP_ID: Instance[BBFP_ID] = Instantiate(new BBFP_ID(parameter))
  BBFP_ID.io.cmdReq <> io.cmdReq
// -----------------------------------------------------------------------------
// ID_LU
// -----------------------------------------------------------------------------
  val ID_LU:   Instance[ID_LU]   = Instantiate(new ID_LU(parameter))
  ID_LU.io.id_lu_i <> BBFP_ID.io.id_lu_o

// -----------------------------------------------------------------------------
// BBFP_LoadUnit
// -----------------------------------------------------------------------------
  val BBFP_LoadUnit: Instance[BBFP_LoadUnit] = Instantiate(new BBFP_LoadUnit(parameter))
  BBFP_LoadUnit.io.id_lu_i <> ID_LU.io.ld_lu_o
  for (i <- 0 until parameter.numBanks) {
    io.sramRead(i).req <> BBFP_LoadUnit.io.sramReadReq(i)
  }
// -----------------------------------------------------------------------------
// LU_EX
// -----------------------------------------------------------------------------
  val LU_EX: Instance[LU_EX] = Instantiate(new LU_EX(parameter))
  LU_EX.io.lu_ex_i <> BBFP_LoadUnit.io.lu_ex_o

// -----------------------------------------------------------------------------
// BBFP_EX
// -----------------------------------------------------------------------------
  val BBFP_EX: Instance[BBFP_EX] = Instantiate(new BBFP_EX(parameter))
  BBFP_EX.io.lu_ex_i <> LU_EX.io.lu_ex_o
  for (i <- 0 until parameter.numBanks) {
    BBFP_EX.io.sramReadResp(i) <> io.sramRead(i).resp
    io.sramWrite(i) <> BBFP_EX.io.sramWrite(i)
  }
  BBFP_EX.io.is_matmul_ws := io.is_matmul_ws
  for (i <- 0 until parameter.numBanks) {
    io.accWrite(i) <> BBFP_EX.io.accWrite(i)
  }
  io.cmdResp <> BBFP_EX.io.cmdResp

  // Status tracking
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
