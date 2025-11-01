package prototype.vector.warp

import chisel3._
import chisel3.util._
import chisel3.stage._
import org.chipsalliance.cde.config.Parameters

import prototype.vector.thread._
import prototype.vector.bond.BondWrapper

class MeshWarpInput extends Bundle {
  val op1 = Vec(16, UInt(8.W))
  val op2 = Vec(16, UInt(8.W))
  val thread_id = UInt(10.W)
}

class MeshWarpOutput extends Bundle {
  val res = Vec(16, UInt(32.W))
}

class MeshWarp(implicit p: Parameters) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(Decoupled(new MeshWarpInput))
    val out = Decoupled(new MeshWarpOutput)
  })

  val threadMap = (0 until 32).map { i =>
    val threadName = i.toString
    // 0-15 mul, 16-31 cascade
    val opType = if (i < 16) "mul" else "cascade"
    // mul operation: 8-bit input, 32-bit output; cascade operation: 32-bit input, 32-bit output
    val bond = if (opType == "mul") {
      BondParam("vvv", inputWidth = 8, outputWidth = 32)
    } else {
      BondParam("vvv", inputWidth = 32, outputWidth = 32)
    }
    val op = OpParam(opType, bond)
    // All threads use the same lane count
    val thread = ThreadParam(16, s"attr$threadName", threadName, op)
    threadName -> thread
  }.toMap

  val mulThreads = (0 until 16).map { i =>
    val threadName = i.toString
    val threadParam = threadMap(threadName)
    val opParam = threadParam.Op
    val bondParam = opParam.bondType
    val threadParams = p.alterMap(Map(
      ThreadMapKey -> threadMap,
      ThreadKey -> Some(threadParam),
      ThreadOpKey -> Some(opParam),
      ThreadBondKey -> Some(bondParam)
    ))
    Module(new MulThread()(threadParams))
  }

  val casThreads = (16 until 32).map { i =>
    val threadName = i.toString
    val threadParam = threadMap(threadName)
    val opParam = threadParam.Op
    val bondParam = opParam.bondType
    val threadParams = p.alterMap(Map(
      ThreadMapKey -> threadMap,
      ThreadKey -> Some(threadParam),
      ThreadOpKey -> Some(opParam),
      ThreadBondKey -> Some(bondParam)
    ))
    Module(new CasThread()(threadParams))
  }

  io.in.ready := mulThreads(0).vvvBond.get.in.ready

  for (i <- 0 until 16) {
   val mulThread = mulThreads(i)
    val casThread = casThreads(i)
    for {
      mulBond <- mulThread.vvvBond
      casBond <- casThread.vvvBond
    } {
      // Connect mul thread output to cascade thread input
      casBond.in.bits.in1 := mulBond.out.bits.out
      mulBond.out.ready   := casBond.in.ready

      // Connect cascade thread's second input and output ready signal
      if (i == 0) {
        casBond.in.bits.in2 := VecInit(Seq.fill(16)(0.U(32.W)))
        // First cascade thread's valid is determined by mulBond's output valid
        casBond.in.valid := mulBond.out.valid
        // First cascade thread's output ready is connected to next cascade thread's input ready
        if (i < 15) {
          for {
            nextCasBond <- casThreads(i + 1).vvvBond
          } {
            casBond.out.ready := nextCasBond.in.ready
          }
        }
      } else {
        for {
          prevCasBond <- casThreads(i - 1).vvvBond
        } {
          // Directly connect 32-bit output to 32-bit input
          casBond.in.bits.in2 := prevCasBond.out.bits.out
          // casBond's valid is jointly determined by previous casBond's output valid and current mulBond's output valid
          casBond.in.valid := prevCasBond.out.valid || mulBond.out.valid
        }
        // Middle cascade thread's output ready is connected to next cascade thread's input ready
        if (i < 15) {
          for {
            nextCasBond <- casThreads(i + 1).vvvBond
          } {
            casBond.out.ready := nextCasBond.in.ready
          }
        }
      }

      // Only allow mulOp corresponding to thread_id to drive input
      when (i.U === io.in.bits.thread_id && io.in.valid) {
        mulBond.in.valid := true.B
        mulBond.in.bits.in1 := io.in.bits.op1
        mulBond.in.bits.in2 := io.in.bits.op2
        io.in.ready := mulBond.in.ready
      }.otherwise {
        mulBond.in.valid := false.B
        mulBond.in.bits.in1 := VecInit(Seq.fill(16)(0.U(8.W)))
        mulBond.in.bits.in2 := VecInit(Seq.fill(16)(0.U(8.W)))
      }
    }
  }

  // Connect output
  for {
    finalCasBond <- casThreads(15).vvvBond
  } {
    io.out.valid := finalCasBond.out.valid
    io.out.bits.res := finalCasBond.out.bits.out
    finalCasBond.out.ready := io.out.ready
  }
}
