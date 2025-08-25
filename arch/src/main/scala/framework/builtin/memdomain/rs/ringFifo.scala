package framework.builtin.memdomain.rs

import chisel3._
import chisel3.util._
import chisel3.experimental._
import chisel3.util._

// this class is unused

class RingFifo[T <: Data](gen: T, n: Int) extends Module {
  require(n > 0, "FIFO size must be greater than 0")

  val io = IO(new Bundle{
    val enq = Flipped(new DecoupledIO(gen)) // Flipped是反转接口
    val deq = new DecoupledIO(gen)
  })
  
  val enqPtr = RegInit(0.U(log2Up(n).W)) // 栈尾
  val deqPtr = RegInit(0.U(log2Up(n).W)) // 栈首
  val isFull = RegInit(false.B)  // 是否满了
  
  val doEnq = io.enq.ready && io.enq.valid // 需要执行入栈，入栈操作开启且入栈的元素有效
  val doDeq = io.deq.ready && io.deq.valid // 执行出栈
  
  val isEmpty = !isFull && (enqPtr === deqPtr) // 栈空
  
  val deqPtrInc = deqPtr + 1.U
  val enqPtrInc = enqPtr + 1.U
  
  // 判断接下来是否会满
  val isFullNext = Mux(doEnq && !doDeq && (enqPtrInc === deqPtr),  // 入栈，且不出栈，且栈接下会满 
                       true.B , Mux(doDeq && isFull, // 要出栈，且满了
                             false.B, isFull))
  enqPtr := Mux(doEnq, enqPtrInc, enqPtr) // 入栈，改变尾，向后加一个元素
  deqPtr := Mux(doDeq, deqPtrInc, deqPtr) // 出栈，改变首，头向后移一个
  
  isFull := isFullNext
  val ram = Mem(n, gen)
  when (doEnq){
    ram(enqPtr) := io.enq.bits
  }
  io.enq.ready := !isFull
  io.deq.valid := !isEmpty
  
  ram(deqPtr) <> io.deq.bits
}

