package framework.memdomain.frontend.cmd_channel.rs

import chisel3._
import chisel3.util._
import chisel3.experimental._
import chisel3.util._

// this class is unused

class RingFifo[T <: Data](gen: T, n: Int) extends Module {
  require(n > 0, "FIFO size must be greater than 0")

  val io = IO(new Bundle {
    // Flipped reverses the interface
    val enq = Flipped(new DecoupledIO(gen))
    val deq = new DecoupledIO(gen)
  })

  // Stack tail
  val enqPtr = RegInit(0.U(log2Up(n).W))
  // Stack head
  val deqPtr = RegInit(0.U(log2Up(n).W))
  // Whether it is full
  val isFull = RegInit(false.B)

  // Need to execute enqueue, enqueue operation is enabled and enqueue element is valid
  val doEnq = io.enq.ready && io.enq.valid
  // Execute dequeue
  val doDeq = io.deq.ready && io.deq.valid

  // Stack empty
  val isEmpty = !isFull && (enqPtr === deqPtr)

  val deqPtrInc = deqPtr + 1.U
  val enqPtrInc = enqPtr + 1.U

  // Determine if it will be full next
  // Enqueue, and no dequeue, and stack will be full next
  val isFullNext = Mux(
    doEnq && !doDeq && (enqPtrInc === deqPtr),
    true.B,
    Mux(
      doDeq && isFull, // Dequeue, and full
      false.B,
      isFull
    )
  )

  // Enqueue, change tail, add one element backward
  enqPtr := Mux(doEnq, enqPtrInc, enqPtr)
  // Dequeue, change head, head moves backward by one
  deqPtr := Mux(doDeq, deqPtrInc, deqPtr)

  isFull := isFullNext
  val ram = Mem(n, gen)
  when(doEnq) {
    ram(enqPtr) := io.enq.bits
  }
  io.enq.ready := !isFull
  io.deq.valid := !isEmpty

  ram(deqPtr) <> io.deq.bits
}
