package framework.balldomain.prototype.vector.warp

import chisel3._
import chisel3.util._
import chisel3.stage._
import framework.balldomain.prototype.vector.configs.VectorBallParam
import framework.balldomain.prototype.vector.thread._

class MeshWarpInput(val config: VectorBallParam) extends Bundle {
  val op1       = Vec(config.lane, UInt(config.inputWidth.W))
  val op2       = Vec(config.lane, UInt(config.inputWidth.W))
  val thread_id = UInt(10.W)
}

class MeshWarpOutput(val config: VectorBallParam) extends Bundle {
  val res = Vec(config.lane, UInt(config.outputWidth.W))
}

class MeshWarp(val config: VectorBallParam) extends Module {

  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(new MeshWarpInput(config)))
    val out = Decoupled(new MeshWarpOutput(config))
  })

  val mulThreads = (0 until config.numMulThreads).map { i =>
    Module(new MulThread(config))
  }

  val casThreads = (0 until config.numCasThreads).map { i =>
    Module(new CasThread(config))
  }

  io.in.ready := mulThreads(0).vvvBond.in.ready

  for (i <- 0 until config.numMulThreads) {
    val mulThread = mulThreads(i)
    val casThread = casThreads(i)
    val mulBond   = mulThread.vvvBond
    val casBond   = casThread.vvvBond

    // Connect mul thread output to cascade thread input
    casBond.in.bits.in1 := mulBond.out.bits.out
    mulBond.out.ready   := casBond.in.ready

    // Connect cascade thread's second input and output ready signal
    if (i == 0) {
      casBond.in.bits.in2 := VecInit(Seq.fill(config.lane)(0.U(config.outputWidth.W)))
      // First cascade thread's valid is determined by mulBond's output valid
      casBond.in.valid    := mulBond.out.valid
      // First cascade thread's output ready is connected to next cascade thread's input ready
      if (i < config.numCasThreads - 1) {
        casBond.out.ready := casThreads(i + 1).vvvBond.in.ready
      }
    } else {
      // Directly connect output to input
      casBond.in.bits.in2 := casThreads(i - 1).vvvBond.out.bits.out
      // casBond's valid is jointly determined by previous casBond's output valid and current mulBond's output valid
      casBond.in.valid    := casThreads(i - 1).vvvBond.out.valid || mulBond.out.valid
      // Middle cascade thread's output ready is connected to next cascade thread's input ready
      if (i < config.numCasThreads - 1) {
        casBond.out.ready := casThreads(i + 1).vvvBond.in.ready
      }
    }

    // Only allow mulOp corresponding to thread_id to drive input
    when(i.U === io.in.bits.thread_id && io.in.valid) {
      mulBond.in.valid    := true.B
      mulBond.in.bits.in1 := io.in.bits.op1
      mulBond.in.bits.in2 := io.in.bits.op2
      io.in.ready         := mulBond.in.ready
    }.otherwise {
      mulBond.in.valid    := false.B
      mulBond.in.bits.in1 := VecInit(Seq.fill(config.lane)(0.U(config.inputWidth.W)))
      mulBond.in.bits.in2 := VecInit(Seq.fill(config.lane)(0.U(config.inputWidth.W)))
    }
  }

  // Connect output
  val finalCasBond = casThreads(config.numCasThreads - 1).vvvBond
  io.out.valid           := finalCasBond.out.valid
  io.out.bits.res        := finalCasBond.out.bits.out
  finalCasBond.out.ready := io.out.ready
}
