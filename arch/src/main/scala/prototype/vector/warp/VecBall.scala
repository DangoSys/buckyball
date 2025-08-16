package prototype.vector.warp

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._

class BallIO extends Bundle {
  // val start = Output(Bool())
  // val arrive = Output(Bool())
  // val done = Output(Bool())
  val iterIn = Flipped(Decoupled(UInt(10.W)))
  val iterOut = Valid(UInt(10.W))
}

class VecBallIO extends BallIO {
  val op1In = Flipped(Valid(Vec(16, UInt(8.W))))
  val op2In = Flipped(Valid(Vec(16, UInt(8.W))))
  val rstOut = Decoupled(Vec(16, UInt(32.W)))
}

class VecBall(implicit p: Parameters) extends Module {
  val io = IO(new VecBallIO())

  // 内部状态寄存器 & 迭代计数器
  val start  = RegInit(false.B)
  val arrive = RegInit(false.B)
  val done   = RegInit(false.B)
  val iter   = RegInit(0.U(10.W))
  val iterCounter = RegInit(0.U(10.W))

  // 单独控制逻辑
  val threadId = RegInit(0.U(4.W))
  when (io.op1In.valid && io.op2In.valid && threadId < 15.U) {
    threadId := threadId + 1.U
  } .elsewhen (io.op1In.valid && io.op2In.valid && threadId === 15.U) {
    threadId := 0.U
  }

  // 实例化MeshWarp
  val meshWarp = Module(new MeshWarp()(p))

  // 连接外部IO到MeshWarp
  meshWarp.io.in.valid := io.op1In.valid && io.op2In.valid
  meshWarp.io.in.bits.op1 := io.op1In.bits
  meshWarp.io.in.bits.op2 := io.op2In.bits
  meshWarp.io.in.bits.thread_id := threadId

  io.rstOut.valid := meshWarp.io.out.valid
  io.rstOut.bits := meshWarp.io.out.bits.res
  meshWarp.io.out.ready := io.rstOut.ready

  // 处理迭代输入
  when (io.iterIn.fire) {iterCounter := 0.U; iter := io.iterIn.bits}
  // 当外部输入来临时start拉高
  when (io.op1In.valid && io.op2In.valid) {start := true.B}
  // 当第一个输出开始valid后arrive拉高
  when (io.rstOut.valid && !arrive) {arrive := true.B}
  // 每出来一个数就加一
  when (io.rstOut.valid && iterCounter =/= iter) {iterCounter := iterCounter + 1.U}
  // 当iter回到0后就done拉高
  when (iterCounter === iter) {done := true.B}

  // 重置逻辑
  when (io.iterIn.fire) {
    start   := false.B
    arrive  := false.B
    done    := false.B
    iterCounter := 0.U
  }

  // 输出状态
  // io.start := start
  // io.arrive := arrive
  // io.done := done

  // 输出当前迭代计数
  io.iterOut.valid := io.rstOut.valid
  io.iterOut.bits := iterCounter
  io.iterIn.ready := meshWarp.io.in.ready

  // def get_iterCounter(): UInt = {
  //   iterCounter
  // }

  // def get_arrive(): Bool = {
  //   arrive
  // }

}