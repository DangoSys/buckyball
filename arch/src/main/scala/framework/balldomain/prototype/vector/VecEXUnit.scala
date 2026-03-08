package framework.balldomain.prototype.vector

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig
import framework.balldomain.prototype.vector.warp.MeshWarp
import framework.balldomain.prototype.vector.configs.VectorBallParam

class ctrl_ex_req(b: GlobalConfig) extends Bundle {
  val iter = UInt(b.frontend.iter_len.W)
}

class ld_ex_req(b: GlobalConfig) extends Bundle {
  val config = VectorBallParam()
  val op1    = Vec(config.lane, UInt(config.inputWidth.W))
  val op2    = Vec(config.lane, UInt(config.inputWidth.W))
  val iter   = UInt(b.frontend.iter_len.W)
}

@instantiable
class VecEXUnit(val b: GlobalConfig) extends Module {
  val config     = VectorBallParam()
  val InputNum   = config.lane
  val inputWidth = config.inputWidth
  val accWidth   = config.outputWidth

  @public
  val io = IO(new Bundle {
    val ctrl_ex_i = Flipped(Decoupled(new ctrl_ex_req(b)))
    val ld_ex_i   = Flipped(Decoupled(new ld_ex_req(b)))

    val ex_st_o = Decoupled(new ex_st_req(b))
  })

  val idle :: busy :: Nil = Enum(2)
  val state               = RegInit(idle)

  val meshWarp = Module(new MeshWarp(config))

  // Thread ID for MeshWarp (always use thread 0 for now)
  val threadId = RegInit(0.U(10.W))

  // Initialize default values for all signals
  io.ctrl_ex_i.ready   := false.B
  io.ex_st_o.valid     := false.B
  io.ex_st_o.bits.rst  := VecInit(Seq.fill(InputNum)(0.U(accWidth.W)))
  io.ex_st_o.bits.iter := 0.U

  // Initialize MeshWarp input signals with default values
  meshWarp.io.in.valid          := false.B
  meshWarp.io.in.bits.op1       := VecInit(Seq.fill(InputNum)(0.U(inputWidth.W)))
  meshWarp.io.in.bits.op2       := VecInit(Seq.fill(InputNum)(0.U(inputWidth.W)))
  meshWarp.io.in.bits.thread_id := threadId
  meshWarp.io.out.ready         := false.B

// -----------------------------------------------------------------------------
// Set registers when Ctrl instruction arrives
// -----------------------------------------------------------------------------
  io.ctrl_ex_i.ready := state === idle
  when(io.ctrl_ex_i.fire) {
    threadId := 0.U // Use thread 0 for computation
    state    := busy
  }
  when(io.ld_ex_i.fire) {
    threadId := Mux(threadId === (config.numMulThreads - 1).U, 0.U, threadId + 1.U)
  }

// -----------------------------------------------------------------------------
// Accept read results from load unit and perform computation
// -----------------------------------------------------------------------------
  io.ld_ex_i.ready := state === busy && meshWarp.io.in.ready
  when(io.ld_ex_i.valid) {
    meshWarp.io.in.valid          := true.B
    meshWarp.io.in.bits.op1       := io.ld_ex_i.bits.op1
    meshWarp.io.in.bits.op2       := io.ld_ex_i.bits.op2
    meshWarp.io.in.bits.thread_id := threadId
  }

// -----------------------------------------------------------------------------
// Send computation results to store unit for write-back
// -----------------------------------------------------------------------------
  io.ex_st_o.valid      := meshWarp.io.out.valid
  meshWarp.io.out.ready := io.ex_st_o.ready

  when(io.ex_st_o.fire) {
    io.ex_st_o.bits.rst  := meshWarp.io.out.bits.res
    io.ex_st_o.bits.iter := io.ld_ex_i.bits.iter // Use iter from ld_ex_i
  }

}
