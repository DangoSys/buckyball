package prototype.vector.warp

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._

class BallIO extends Bundle {
  // val start = Output(Bool())
  // val arrive = Output(Bool())
  // val done = Output(Bool())
  val iterIn  = Flipped(Decoupled(UInt(10.W)))
  val iterOut = Valid(UInt(10.W))
}

class VecBallIO extends BallIO {
  val op1In  = Flipped(Valid(Vec(16, UInt(8.W))))
  val op2In  = Flipped(Valid(Vec(16, UInt(8.W))))
  val rstOut = Decoupled(Vec(16, UInt(32.W)))
}

class VecBall(implicit p: Parameters) extends Module {
  val io = IO(new VecBallIO())

  // Internal state registers & iteration counter
  val start       = RegInit(false.B)
  val arrive      = RegInit(false.B)
  val done        = RegInit(false.B)
  val iter        = RegInit(0.U(10.W))
  val iterCounter = RegInit(0.U(10.W))

  // Independent control logic
  val threadId = RegInit(0.U(4.W))
  when(io.op1In.valid && io.op2In.valid && threadId < 15.U) {
    threadId := threadId + 1.U
  }.elsewhen(io.op1In.valid && io.op2In.valid && threadId === 15.U) {
    threadId := 0.U
  }

  // Instantiate MeshWarp
  val meshWarp = Module(new MeshWarp()(p))

  // Connect external IO to MeshWarp
  meshWarp.io.in.valid          := io.op1In.valid && io.op2In.valid
  meshWarp.io.in.bits.op1       := io.op1In.bits
  meshWarp.io.in.bits.op2       := io.op2In.bits
  meshWarp.io.in.bits.thread_id := threadId

  io.rstOut.valid       := meshWarp.io.out.valid
  io.rstOut.bits        := meshWarp.io.out.bits.res
  meshWarp.io.out.ready := io.rstOut.ready

  // Handle iteration input
  when(io.iterIn.fire) { iterCounter := 0.U; iter := io.iterIn.bits }
  // Pull start high when external input arrives
  when(io.op1In.valid && io.op2In.valid)(start              := true.B)
  // Pull arrive high when first output starts to be valid
  when(io.rstOut.valid && !arrive)(arrive                   := true.B)
  // Increment by one for each output
  when(io.rstOut.valid && iterCounter =/= iter)(iterCounter := iterCounter + 1.U)
  // Pull done high when iter returns to 0
  when(iterCounter === iter)(done                           := true.B)

  // Reset logic
  when(io.iterIn.fire) {
    start       := false.B
    arrive      := false.B
    done        := false.B
    iterCounter := 0.U
  }

  // Output state
  // io.start := start
  // io.arrive := arrive
  // io.done := done

  // Output current iteration count
  io.iterOut.valid := io.rstOut.valid
  io.iterOut.bits  := iterCounter
  io.iterIn.ready  := meshWarp.io.in.ready

  // def get_iterCounter(): UInt = {
  //   iterCounter
  // }

  // def get_arrive(): Bool = {
  //   arrive
  // }

}
