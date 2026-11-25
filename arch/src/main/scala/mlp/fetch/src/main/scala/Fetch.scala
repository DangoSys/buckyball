package framework.mlp.fetch

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.mlp.fetch.configs.{Params => ModuleParam}

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
  val data           = UInt(64.W)
  val line           = UInt(512.W)
  val lineExecutable = Bool()
  val pageFault      = Bool()
  val accessFault    = Bool()
}

class Packet(b: ModuleParam) extends Bundle {
  val p = b

  val pc          = UInt(64.W)
  val data        = UInt(32.W)
  val mask        = UInt(p.lanes.W)
  val pageFault   = Bool()
  val accessFault = Bool()
}

/** One I$ line: its virtual tag, the fetch privilege and satp it was translated under, and its data. */
class Line(tagBits: Int) extends Bundle {
  val tag       = UInt(tagBits.W)
  val privilege = UInt(2.W)
  val satp      = UInt(64.W)
  val data      = UInt(512.W)
}

/**
 * Instruction fetch with a direct-mapped, virtually tagged L1 I$ in one single-port SRAM. A lookup
 * registers the SRAM result before checking its tag. The line stays held, so the following
 * packets of that line need no read. The read of the next line starts while delivering the last
 * packet of the current one. A miss fetches the doubleword
 * through the memory system and fills the whole line when the line is executable. Lines are keyed
 * by virtual line, privilege and satp, and are dropped by fence.i (`flush`) and sfence.vma
 * (`invalidate`).
 */
@instantiable
class Fetch(b: ModuleParam) extends Module {
  val p = b

  @public
  val io = IO(new Bundle {
    val resetVector = Input(UInt(64.W))
    val context     = Input(new Context)
    val redirect    = Flipped(Valid(UInt(64.W)))
    val flush       = Input(Bool())
    val invalidate  = Input(Bool())
    val packet      = Decoupled(new Packet(b))
    val request     = Decoupled(new Request)
    val response    = Flipped(Decoupled(new Response))
    val maintenance = Decoupled(Bool())
    val maintained  = Flipped(Decoupled(Bool()))
    val npc         = Output(UInt(64.W))
  })

  val indexBits = log2Ceil(p.lines)
  val tagBits   = 58 - indexBits
  def index(addr: UInt): UInt = addr(indexBits + 5, 6)
  def tag(addr:   UInt): UInt = addr(63, indexBits + 6)

  val run :: offer :: receive :: deliver :: maintain :: waitMaintenance :: Nil = Enum(6)
  val state                                                                    = RegInit(run)
  val pc                                                                       = Reg(UInt(64.W))
  val readPc                                                                   = Reg(UInt(64.W))
  val command                                                                  = Reg(new Request)
  val answer                                                                   = Reg(new Packet(b))
  val discard                                                                  = RegInit(false.B)
  val pendingMaintenance                                                       = RegInit(false.B)
  // A line read before fence.i or sfence.vma must not fill the I$ after them.
  val staleFill                                                                = RegInit(false.B)
  val cancelled                                                                = io.redirect.valid || io.flush

  val valid     = RegInit(VecInit(Seq.fill(p.lines)(false.B)))
  val array     = SyncReadMem(p.lines, new Line(tagBits))
  val read      = WireDefault(false.B)
  val write     = WireDefault(false.B)
  val address   = WireDefault(0.U(indexBits.W))
  val writeLine = WireDefault(0.U.asTypeOf(new Line(tagBits)))
  val raw       = array.readWrite(address, writeLine, read || write, write)
  val readDone  = RegNext(read, false.B)
  val lookupPc  = Reg(UInt(64.W))
  // Keep the SRAM output out of the feedback path to its next address and enable.
  val stored    = RegEnable(raw, readDone)
  val fresh     = RegNext(readDone, false.B)
  val lookPc    = RegEnable(lookupPc, readDone)

  // The line the following packets are taken from, without another SRAM read.
  val held          = RegInit(false.B)
  val heldLine      = Reg(UInt(58.W))
  val heldPrivilege = Reg(UInt(2.W))
  val heldSatp      = Reg(UInt(64.W))
  val heldData      = Reg(UInt(512.W))

  def packetAt(addr: UInt, data: UInt): Packet = {
    val packet = Wire(new Packet(b))
    packet.pc          := addr
    packet.data        := (data >> (addr(5, 2) << 5))(31, 0)
    packet.mask        := (if (p.instructionBytes == 2) (3.U(2.W) << addr(1))(1, 0) else 1.U)
    packet.pageFault   := false.B
    packet.accessFault := false.B
    packet
  }

  val sameContext = heldPrivilege === io.context.privilege && heldSatp === io.context.satp
  val heldHit     = held && heldLine === pc(63, 6) && sameContext
  val looked      = state === run && fresh && lookPc === pc
  val freshHit    = looked && valid(index(pc)) && stored.tag === tag(pc) &&
    stored.privilege === io.context.privilege && stored.satp === io.context.satp
  val serving     = state === run && !pendingMaintenance && (heldHit || freshHit)
  val miss        = looked && !freshHit && !heldHit && !pendingMaintenance

  // A miss is requested in the cycle its lookup misses; a filled line delivers its packet as it returns.
  val missing  = miss && !cancelled
  val response = io.response.bits
  val arriving = state === receive && io.response.valid && !discard && !staleFill &&
    !response.pageFault && !response.accessFault && response.lineExecutable
  io.npc                  := Mux(io.redirect.valid, io.redirect.bits, pc)
  io.request.valid        := state === offer || missing
  io.request.bits         := command
  when(state === run) {
    io.request.bits.addr    := pc & ~7.U(64.W)
    io.request.bits.context := io.context
  }
  io.request.bits.execute := true.B
  io.response.ready       := state === receive
  // A redirecting cycle delivers nothing: its packet would be of the old path.
  io.packet.valid         := (serving || state === deliver || arriving) && !cancelled
  io.packet.bits          := Mux(
    state === deliver,
    answer,
    packetAt(pc, Mux(arriving, response.line, Mux(heldHit, heldData, stored.data)))
  )
  io.maintenance.valid    := state === maintain
  io.maintenance.bits     := true.B
  io.maintained.ready     := state === waitMaintenance

  when(freshHit && !heldHit) {
    held          := true.B
    heldLine      := pc(63, 6)
    heldPrivilege := io.context.privilege
    heldSatp      := io.context.satp
    heldData      := stored.data
  }

  // The PC that needs a packet in the next cycle; read its line unless that line will be held.
  val next     = (pc & ~3.U(64.W)) + 4.U
  val wanted   = Mux(io.redirect.valid, io.redirect.bits, Mux(io.packet.fire && state === run, next, pc))
  val nextHeld = freshHit || held && sameContext
  val nextLine = Mux(freshHit && !heldHit, pc(63, 6), heldLine)
  val covered  = nextHeld && nextLine === wanted(63, 6)
  read                := state === run && !pendingMaintenance && !(miss && !io.redirect.valid) && !io.flush && !covered
  address             := index(wanted)
  when(read)(lookupPc := wanted)

  when(io.packet.fire) {
    pc                            := next
    when(state === deliver)(state := run)
  }

  when(missing) {
    assert(io.context.privilege =/= 2.U, "Fetch privilege must be U, S or M")
    assert(pc(log2Ceil(p.instructionBytes) - 1, 0) === 0.U, "Fetch reset vector must be instruction aligned")
    command.addr    := pc & ~7.U(64.W)
    command.context := io.context
    readPc          := pc
    discard         := false.B
    staleFill       := false.B
    state           := offer
  }
  when(state === run && pendingMaintenance && !cancelled)(state := maintain)
  when(io.request.fire)(state                                   := receive)
  when(io.response.fire) {
    val fault = io.response.bits.pageFault || io.response.bits.accessFault
    val fills = !fault && io.response.bits.lineExecutable && !staleFill && !io.flush && !io.invalidate
    when(fills) {
      write                      := true.B
      address                    := index(command.addr)
      writeLine.tag              := tag(command.addr)
      writeLine.privilege        := command.context.privilege
      writeLine.satp             := command.context.satp
      writeLine.data             := io.response.bits.line
      valid(index(command.addr)) := true.B
      held                       := true.B
      heldLine                   := command.addr(63, 6)
      heldPrivilege              := command.context.privilege
      heldSatp                   := command.context.satp
      heldData                   := io.response.bits.line
    }
    when(discard || cancelled || fills) {
      state := run
    }.otherwise {
      answer             := packetAt(readPc, 0.U(512.W))
      answer.data        := Mux(fault, 0.U, (io.response.bits.data >> (readPc(2) << 5))(31, 0))
      answer.pageFault   := io.response.bits.pageFault
      answer.accessFault := io.response.bits.accessFault
      state              := deliver
    }
  }
  when(io.maintenance.fire) {
    pendingMaintenance := false.B
    state              := waitMaintenance
  }
  when(io.maintained.fire) {
    assert(io.maintained.bits, "Fetch maintenance response must acknowledge completion")
    state := run
  }
  when(io.redirect.valid) {
    assert(io.redirect.bits(log2Ceil(p.instructionBytes) - 1, 0) === 0.U, "Fetch redirect must be instruction aligned")
    pc                            := io.redirect.bits
    discard                       := true.B
    when(state === deliver)(state := run)
  }
  when(io.flush || io.invalidate) {
    valid.foreach(_ := false.B)
    held            := false.B
    staleFill       := true.B
  }
  when(io.flush) {
    assert(io.redirect.valid, "Fetch flush requires a restart redirect")
    pendingMaintenance := true.B
  }
  when(reset.asBool) {
    pc := io.resetVector
  }
}
