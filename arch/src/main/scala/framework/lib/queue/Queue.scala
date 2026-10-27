package framework.lib.queue

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.util.{log2Ceil, QueueIO}

@instantiable
class Queue[T <: Data](
  gen:     T,
  entries: Int,
  pipe:    Boolean = false,
  flow:    Boolean = false)
    extends Module {
  require(entries > 0)
  @public
  val io = IO(new QueueIO(gen, entries))

  val count       = RegInit(0.U(log2Ceil(entries + 1).W))
  val memoryCount = RegInit(0.U(log2Ceil(entries + 1).W))
  val front       = Reg(gen)
  val frontValid  = RegInit(false.B)
  val skid        = Reg(gen)
  val skidValid   = RegInit(false.B)
  val readPending = RegInit(false.B)

  io.count     := count
  io.deq.valid := frontValid || (if (flow) count === 0.U && io.enq.valid else false.B)
  io.deq.bits  := (if (flow) Mux(frontValid, front, io.enq.bits) else front)
  val dequeue = frontValid && io.deq.ready
  io.enq.ready := count < entries.U || (if (pipe) dequeue else false.B)
  val bypass  = if (flow) count === 0.U && io.deq.fire else false.B
  val enqueue = io.enq.fire && !bypass
  val direct  = enqueue && memoryCount === 0.U && !readPending && !skidValid && (!frontValid || dequeue)
  val write   = enqueue && !direct

  val nextFront      = WireDefault(front)
  val nextFrontValid = WireDefault(frontValid && !dequeue)
  val nextSkid       = WireDefault(skid)
  val nextSkidValid  = WireDefault(skidValid)
  when((!frontValid || dequeue) && skidValid) {
    nextFront      := skid
    nextFrontValid := true.B
    nextSkidValid  := false.B
  }
  if (entries > 2) {
    val memory   = SyncReadMem(entries, gen)
    val head     = RegInit(0.U(log2Ceil(entries).W))
    val tail     = RegInit(0.U(log2Ceil(entries).W))
    val occupied = frontValid.asUInt +& skidValid.asUInt +& readPending.asUInt
    val read     = memoryCount =/= 0.U && !write && occupied < (2.U + dequeue.asUInt)
    val readData = memory.readWrite(Mux(write, tail, head), io.enq.bits, write || read, write)
    when(readPending) {
      when((!frontValid || dequeue) && !skidValid) {
        nextFront      := readData
        nextFrontValid := true.B
      }.otherwise {
        nextSkid      := readData
        nextSkidValid := true.B
      }
    }
    readPending := read
    when(write =/= read) {
      memoryCount := Mux(write, memoryCount + 1.U, memoryCount - 1.U)
    }
    when(write)(tail := Mux(tail === (entries - 1).U, 0.U, tail + 1.U))
    when(read)(head := Mux(head === (entries - 1).U, 0.U, head + 1.U))
  } else {
    // Front and skid registers hold up to two entries, so small queues need no memory
    when(write) {
      nextSkid      := io.enq.bits
      nextSkidValid := true.B
    }
  }
  when(direct) {
    nextFront      := io.enq.bits
    nextFrontValid := true.B
  }
  front := nextFront
  frontValid := nextFrontValid
  skid       := nextSkid
  skidValid  := nextSkidValid

  when(enqueue =/= dequeue) {
    count := Mux(enqueue, count + 1.U, count - 1.U)
  }
}
