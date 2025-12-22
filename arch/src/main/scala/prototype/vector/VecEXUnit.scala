package prototype.vector

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}
import org.chipsalliance.cde.config.Parameters

import prototype.vector._
import examples.toy.balldomain.BallDomainParam
import prototype.vector.warp.VecBall

class ctrl_ex_req extends Bundle {
  val iter = UInt(10.W)
}

class ld_ex_req(parameter: BallDomainParam) extends Bundle {
  // Derived parameters
  val InputNum   = 16
  val inputWidth = 8
  val op1        = Vec(InputNum, UInt(inputWidth.W))
  val op2        = Vec(InputNum, UInt(inputWidth.W))
  val iter       = UInt(10.W)
}

@instantiable
class VecEXUnit(val parameter: BallDomainParam)(implicit p: Parameters)
    extends Module
    with SerializableModule[BallDomainParam] {
  // Derived parameters
  val InputNum   = 16
  val inputWidth = 8
  val accWidth   = 32

  @public
  val io = IO(new Bundle {
    val ctrl_ex_i = Flipped(Decoupled(new ctrl_ex_req))
    val ld_ex_i   = Flipped(Decoupled(new ld_ex_req(parameter)))

    val ex_st_o = Decoupled(new ex_st_req(parameter))
  })

  val idle :: busy :: Nil = Enum(2)
  val state               = RegInit(idle)

  val VecBall = Module(new VecBall()(p))

  // Initialize default values for all signals
  io.ctrl_ex_i.ready   := false.B
  io.ex_st_o.valid     := false.B
  io.ex_st_o.bits.rst  := VecInit(Seq.fill(InputNum)(0.U(accWidth.W)))
  io.ex_st_o.bits.iter := 0.U

  // Initialize VecBall input signals with default values
  VecBall.io.iterIn.valid := false.B
  VecBall.io.iterIn.bits  := 0.U
  VecBall.io.op1In.valid  := false.B
  VecBall.io.op1In.bits   := VecInit(Seq.fill(InputNum)(0.U(inputWidth.W)))
  VecBall.io.op2In.valid  := false.B
  VecBall.io.op2In.bits   := VecInit(Seq.fill(InputNum)(0.U(inputWidth.W)))
  VecBall.io.rstOut.ready := false.B

// -----------------------------------------------------------------------------
// Set registers when Ctrl instruction arrives
// -----------------------------------------------------------------------------
  io.ctrl_ex_i.ready := state === idle
  when(io.ctrl_ex_i.fire) {
    VecBall.io.iterIn.valid := true.B
    VecBall.io.iterIn.bits  := io.ctrl_ex_i.bits.iter
    state                   := busy
  }

// -----------------------------------------------------------------------------
// Accept read results from load unit and perform computation
// -----------------------------------------------------------------------------
  io.ld_ex_i.ready := state === busy && VecBall.io.iterIn.ready
  when(io.ld_ex_i.valid) {
    VecBall.io.op1In.valid := true.B
    VecBall.io.op1In.bits  := io.ld_ex_i.bits.op1
    VecBall.io.op2In.valid := true.B
    VecBall.io.op2In.bits  := io.ld_ex_i.bits.op2
    //assert((io.ld_ex_i.bits.iter - VecBall.get_iterCounter() === 16.U) && VecBall.get_arrive(),
    //"[VecLoad -> VecEX] iteration mismatch")
  }

// -----------------------------------------------------------------------------
// Send computation results to store unit for write-back
// -----------------------------------------------------------------------------
  io.ex_st_o.valid        := VecBall.io.rstOut.valid
  VecBall.io.rstOut.ready := io.ex_st_o.ready

  when(io.ex_st_o.fire) {
    io.ex_st_o.bits.rst  := VecBall.io.rstOut.bits
    io.ex_st_o.bits.iter := VecBall.io.iterOut.bits
  }

}
