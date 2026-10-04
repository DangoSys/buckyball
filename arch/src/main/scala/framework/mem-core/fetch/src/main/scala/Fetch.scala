package memcore.memory.fetch

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

case class Params(instructionBytes: Int = 2) {
  require(instructionBytes == 2 || instructionBytes == 4)
  val lanes: Int = 4 / instructionBytes
}

class Context extends Bundle {
  val privilege = UInt(2.W)
  val satp      = UInt(64.W)
  val sum       = Bool()
  val mxr       = Bool()
}

class Request extends Bundle {
  val addr    = UInt(64.W)
  val context = new Context
  val execute = Bool()
}

/** `line` is the whole 64-byte line holding the requested doubleword. */
class Response extends Bundle {
  val data        = UInt(64.W)
  val line        = UInt(512.W)
  val pageFault   = Bool()
  val accessFault = Bool()
}

class Packet(p: Params) extends Bundle {
  val pc          = UInt(64.W)
  val data        = UInt(32.W)
  val mask        = UInt(p.lanes.W)
  val pageFault   = Bool()
  val accessFault = Bool()
}

/**
 * Instruction fetch with two 64-byte line buffers, so a loop crossing one line boundary stays
 * buffered. Packets inside a buffered line are delivered back to back; only a miss in both issues
 * a request, which replaces the less recently used line. Lines are keyed by virtual line and fetch
 * context, and are dropped by fence.i (`flush`) and sfence.vma (`invalidate`).
 */
@instantiable
class Fetch(p: Params) extends Module {

  @public
  val io = IO(new Bundle {
    val resetVector = Input(UInt(64.W))
    val context     = Input(new Context)
    val redirect    = Flipped(Valid(UInt(64.W)))
    val flush       = Input(Bool())
    val invalidate  = Input(Bool())
    val packet      = Decoupled(new Packet(p))
    val request     = Decoupled(new Request)
    val response    = Flipped(Decoupled(new Response))
    val maintenance = Decoupled(Bool())
    val maintained  = Flipped(Decoupled(Bool()))
    val npc         = Output(UInt(64.W))
  })

  val idle :: offer :: receive :: deliver :: maintain :: waitMaintenance :: Nil = Enum(6)
  val state                                                                     = RegInit(idle)
  val pc                                                                        = Reg(UInt(64.W))
  val readPc                                                                    = Reg(UInt(64.W))
  val command                                                                   = Reg(new Request)
  val answer                                                                    = Reg(new Packet(p))
  val discard                                                                   = RegInit(false.B)
  val pendingMaintenance                                                        = RegInit(false.B)
  val cancelled                                                                 = io.redirect.valid || io.flush

  val bufferValid   = RegInit(VecInit(Seq.fill(2)(false.B)))
  val bufferLine    = Reg(Vec(2, UInt(58.W)))
  val bufferData    = Reg(Vec(2, UInt(512.W)))
  val bufferContext = Reg(Vec(2, new Context))
  val recent        = RegInit(0.U(1.W))
  // A line read before fence.i or sfence.vma must not refill the buffer after them.
  val staleFill     = RegInit(false.B)

  def hits(addr: UInt): Vec[Bool] = VecInit((0 until 2).map { i =>
    bufferValid(i) && bufferLine(i) === addr(63, 6) && bufferContext(i).asUInt === io.context.asUInt
  })

  def buffered(addr:     UInt): Bool = hits(addr).asUInt.orR
  def bufferedLine(addr: UInt): UInt = Mux(hits(addr)(1), bufferData(1), bufferData(0))

  def packetAt(addr: UInt, data: UInt): Packet = {
    val packet = Wire(new Packet(p))
    packet.pc          := addr
    packet.data        := (data >> (addr(5, 2) << 5))(31, 0)
    packet.mask        := (if (p.instructionBytes == 2) (3.U(2.W) << addr(1))(1, 0) else 1.U)
    packet.pageFault   := false.B
    packet.accessFault := false.B
    packet
  }

  io.npc                  := Mux(io.redirect.valid, io.redirect.bits, pc)
  io.request.valid        := state === offer
  io.request.bits         := command
  io.request.bits.execute := true.B
  io.response.ready       := state === receive
  io.packet.valid         := state === deliver && !cancelled
  io.packet.bits          := answer
  io.maintenance.valid    := state === maintain
  io.maintenance.bits     := true.B
  io.maintained.ready     := state === waitMaintenance

  when(state === idle && !cancelled) {
    when(pendingMaintenance) {
      state := maintain
    }.elsewhen(buffered(pc)) {
      answer := packetAt(pc, bufferedLine(pc))
      readPc := pc
      recent := hits(pc)(1)
      state  := deliver
    }.otherwise {
      assert(io.context.privilege =/= 2.U, "Fetch privilege must be U, S or M")
      assert(pc(log2Ceil(p.instructionBytes) - 1, 0) === 0.U, "Fetch reset vector must be instruction aligned")
      command.addr    := pc & ~7.U(64.W)
      command.context := io.context
      readPc          := pc
      discard         := false.B
      staleFill       := false.B
      state           := offer
    }
  }
  when(io.request.fire)(state := receive)
  when(io.response.fire) {
    val fault = io.response.bits.pageFault || io.response.bits.accessFault
    when(!fault && !staleFill && !io.flush && !io.invalidate) {
      val victim = ~recent
      bufferValid(victim)   := true.B
      bufferLine(victim)    := command.addr(63, 6)
      bufferData(victim)    := io.response.bits.line
      bufferContext(victim) := command.context
      recent                := victim
    }
    when(discard || cancelled) {
      state := idle
    }.otherwise {
      answer             := packetAt(readPc, io.response.bits.line)
      answer.data        := Mux(fault, 0.U, (io.response.bits.line >> (readPc(5, 2) << 5))(31, 0))
      answer.pageFault   := io.response.bits.pageFault
      answer.accessFault := io.response.bits.accessFault
      state              := deliver
    }
  }
  when(io.packet.fire) {
    val next = (readPc & ~3.U(64.W)) + 4.U
    pc := next
    // Faulting packets are never buffered, so a fault always returns through idle.
    when(!answer.pageFault && !answer.accessFault && buffered(next)) {
      answer := packetAt(next, bufferedLine(next))
      readPc := next
      recent := hits(next)(1)
    }.otherwise {
      state := idle
    }
  }
  when(io.maintenance.fire) {
    pendingMaintenance := false.B
    state              := waitMaintenance
  }
  when(io.maintained.fire) {
    assert(io.maintained.bits, "Fetch maintenance response must acknowledge completion")
    state := idle
  }
  when(io.redirect.valid) {
    assert(io.redirect.bits(log2Ceil(p.instructionBytes) - 1, 0) === 0.U, "Fetch redirect must be instruction aligned")
    pc                            := io.redirect.bits
    discard                       := true.B
    when(state === deliver)(state := idle)
  }
  when(io.flush || io.invalidate) {
    bufferValid.foreach(_ := false.B)
    staleFill             := true.B
  }
  when(io.flush) {
    assert(io.redirect.valid, "Fetch flush requires a restart redirect")
    pendingMaintenance := true.B
  }
  when(reset.asBool) {
    pc := io.resetVector
  }
}
