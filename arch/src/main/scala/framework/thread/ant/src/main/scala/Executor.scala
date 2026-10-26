package framework.thread.ant

import framework.thread.ant.configs.AntParam
import framework.core.memory.access.preflight.configs.PmpParams

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.lib.arith.{DivRem, Multiply}
import framework.ip.memory.spm
import framework.thread.ant.{Launch, TaskContext}
import framework.acc.frontend.RoCCIO

class Start(b: AntParam) extends Bundle {
  val p = b

  val task     = UInt(p.taskBits.W)
  val entry    = UInt(64.W)
  val codeEnd  = UInt(64.W)
  val argument = UInt(64.W)
  val stack    = UInt(64.W)
}

class Completion(b: AntParam) extends Bundle {
  val p = b

  val task      = UInt(p.taskBits.W)
  val value     = UInt(64.W)
  val cancelled = Bool()
}

class Retire extends Bundle {
  val pc          = UInt(64.W)
  val instruction = UInt(32.W)
  val rd          = UInt(5.W)
  val data        = UInt(64.W)
}

/**
 * One RV64IM execution context. Ordinary memory is strictly local. NPU work may remain
 * in flight after a command is accepted; completion waits for accelerator and storage drain.
 * Cancellation also reaches the local stores, whose accepted operations must drain before reuse.
 */
@instantiable
class Executor(val b: AntParam, permission: PmpParams) extends Module {
  val p    = b
  val code = spm.Params(0, p.codeBytes, p.data.dataBits)

  @public
  val io = IO(new Bundle {

    val start = Flipped(Decoupled(new Launch(b, permission)))

    val result = Decoupled(new Completion(b))

    val cancel         = Input(Bool())
    val inUse          = Input(Bool())
    val localBusy      = Input(Vec(2, Bool()))
    val storageBusy    = Output(Bool())
    val storageClaimed = Output(Bool())
    val running        = Output(Bool())

    val imem = new spm.Port(code)

    val tls = new spm.Port(p.data)

    val tss = new spm.Port(p.shared)

    val rocc = Flipped(new RoCCIO(64, permission))

    val retired = Output(Valid(new Retire))
  })

  val (idle :: fetch :: fetchWait :: execute :: memory :: memoryWait ::
    multiply :: multiplyWait :: divide :: divideWait :: issue :: complete :: Nil) = Enum(12)
  val state                                                                       = RegInit(idle)
  val registers                                                                   = Reg(Vec(32, UInt(64.W)))
  val pc                                                                          = Reg(UInt(64.W))
  val instruction                                                                 = Reg(UInt(32.W))
  val task                                                                        = Reg(UInt(p.taskBits.W))
  val codeEnd                                                                     = Reg(UInt(64.W))

  val completion = Reg(new Completion(b))
  val context    = Reg(new TaskContext(permission))
  io.storageBusy    := io.localBusy.asUInt.orR
  io.storageClaimed := state =/= idle || (io.inUse && io.start.valid)
  val drained = !io.storageBusy && !io.rocc.busy

  val cancelled       = RegInit(false.B)
  val multiplyPending = RegInit(false.B)
  val dividePending   = RegInit(false.B)
  val live            = !io.cancel && !reset.asBool
  val memoryAddress   = Reg(UInt(64.W))
  val memoryShared    = Reg(Bool())
  val decoder         = Instantiate(new Decode)
  val multiplier      = Instantiate(new Multiply)
  val divider         =
    Instantiate(new DivRem(64))
  decoder.io.instruction := instruction
  val d      = decoder.io.decoded
  val a      = Mux(d.rs1 === 0.U, 0.U, registers(d.rs1))
  val rhs    = Mux(d.rs2 === 0.U, 0.U, registers(d.rs2))
  val nextPc = pc + 4.U

  io.start.ready   := state === idle && io.inUse && drained && !io.cancel && !reset.asBool
  io.result.valid  := state === complete && drained && !multiplyPending && !dividePending && !reset.asBool
  io.result.bits   := completion
  io.running       := state =/= idle
  io.retired.valid := false.B
  io.retired.bits  := 0.U.asTypeOf(new Retire)

  def retire(rd: UInt, value: UInt, next: UInt): Unit = {
    when(live) {
      when(rd =/= 0.U)(registers(rd) := value)
      io.retired.valid               := true.B
      io.retired.bits.pc             := pc
      io.retired.bits.instruction    := instruction
      io.retired.bits.rd             := rd
      io.retired.bits.data           := Mux(rd === 0.U, 0.U, value)
      pc                             := next
      state                          := fetch
    }
  }

  def finish(value: UInt, wasCancelled: Bool = false.B): Unit = {
    completion.task      := task
    completion.value     := value
    completion.cancelled := wasCancelled
    state                := complete
  }

  def word(value: UInt): UInt = Mux(d.word, value(31, 0).asSInt.pad(64).asUInt, value)

  when(io.start.fire) {
    context := io.start.bits.context
    for (r <- registers) r := 0.U
    registers(2)  := io.start.bits.descriptor.stack
    registers(4)  := p.data.base.U
    registers(10) := io.start.bits.descriptor.argument
    pc            := io.start.bits.descriptor.entry
    codeEnd       := io.start.bits.descriptor.codeEnd
    task          := io.start.bits.descriptor.task
    cancelled     := false.B
    val valid = io.start.bits.descriptor.entry(1, 0) === 0.U && io.start.bits.descriptor.codeEnd(1, 0) === 0.U &&
      io.start.bits.descriptor.codeEnd <= p.codeBytes.U && (io.start.bits.descriptor.entry +& 4.U) <= io.start.bits.descriptor.codeEnd &&
      io.start.bits.descriptor.argument >= p.data.base.U && io.start.bits.descriptor.argument < (p.data.base + p.data.bytes).U &&
      io.start.bits.descriptor.stack > p.data.base.U && io.start.bits.descriptor.stack <= (p.data.base + p.data.bytes).U &&
      io.start.bits.descriptor.stack(3, 0) === 0.U
    assert(valid, "Invalid Ant local execution descriptor")
    when(valid)(state := fetch)
  }

  val fetchAllowed = pc(1, 0) === 0.U && (pc +& 4.U) <= codeEnd
  io.imem.request.valid        := state === fetch && fetchAllowed && live
  io.imem.request.bits         := 0.U.asTypeOf(io.imem.request.bits)
  io.imem.request.bits.address := pc
  io.imem.request.bits.size    := 2.U
  io.imem.response.ready       := state === fetchWait || (state === complete && cancelled)
  when(state === fetch && live) {
    assert(fetchAllowed, "Ant instruction address invalid: pc=0x%x", pc)
    when(io.imem.request.fire)(state := fetchWait)
  }
  when(state === fetchWait && io.imem.response.fire && live) {
    assert(!io.imem.response.bits.error, "Ant instruction storage access failed: pc=0x%x", pc)
    when(!io.imem.response.bits.error) {
      instruction := io.imem.response.bits.data(31, 0)
      state       := execute
    }
  }

  val operand = Mux(d.immediateAlu, d.immediate, rhs)
  val shift   = Mux(d.word, Cat(0.U(1.W), operand(4, 0)), operand(5, 0))
  val logical = Mux(d.word, Cat(0.U(32.W), a(31, 0)), a)
  val signed  = Mux(d.word, a(31, 0).asSInt.pad(64).asUInt, a)

  val alu = MuxLookup(d.funct3, 0.U(64.W))(Seq(
    0.U -> Mux(d.subtract, a - operand, a + operand),
    1.U -> (a << shift)(63, 0),
    2.U -> (a.asSInt < operand.asSInt).asUInt,
    3.U -> (a < operand).asUInt,
    4.U -> (a ^ operand),
    5.U -> Mux(d.arithmeticShift, (signed.asSInt >> shift).asUInt, logical >> shift),
    6.U -> (a | operand),
    7.U -> (a & operand)
  ))

  val branchTaken = MuxLookup(d.funct3, false.B)(Seq(
    0.U -> (a === rhs),
    1.U -> (a =/= rhs),
    4.U -> (a.asSInt < rhs.asSInt),
    5.U -> (a.asSInt >= rhs.asSInt),
    6.U -> (a < rhs),
    7.U -> (a >= rhs)
  ))

  val address = a + d.immediate
  val size    = d.funct3(1, 0)
  val bytes   = 1.U(64.W) << size
  val end     = address +& bytes
  val inTls   = address >= p.data.base.U && end <= (p.data.base + p.data.bytes).U
  val inTss   = address >= p.shared.base.U && end <= (p.shared.base + p.shared.bytes).U
  when(state === execute && live) {
    switch(d.kind) {
      is(Kind.illegal)(assert(false.B, "Ant illegal instruction: pc=0x%x instruction=0x%x", pc, instruction))
      is(Kind.lui)(retire(d.rd, d.immediate, nextPc))
      is(Kind.auipc)(retire(d.rd, pc + d.immediate, nextPc))
      is(Kind.alu)(retire(d.rd, word(alu), nextPc))
      is(Kind.jal, Kind.jalr, Kind.branch) {
        val target = Mux(
          d.kind === Kind.jalr,
          (a + d.immediate) & ~1.U(64.W),
          Mux(d.kind === Kind.branch && !branchTaken, nextPc, pc + d.immediate)
        )
        assert(target(1, 0) === 0.U, "Ant branch target misaligned: pc=0x%x target=0x%x", pc, target)
        retire(Mux(d.kind === Kind.branch, 0.U, d.rd), nextPc, target)
      }
      is(Kind.load, Kind.store) {
        assert((address & (bytes - 1.U)) === 0.U, "Ant load/store misaligned: pc=0x%x address=0x%x", pc, address)
        assert(inTls || inTss, "Ant load/store outside TLS/TSS: pc=0x%x address=0x%x", pc, address)
        when((address & (bytes - 1.U)) === 0.U && (inTls || inTss)) {
          memoryAddress := address
          memoryShared  := inTss
          state         := memory
        }
      }
      is(Kind.multiply)(state := multiply)
      is(Kind.divide)(state   := divide)
      is(Kind.custom)(state   := issue)
      is(Kind.fence)(retire(0.U, 0.U, nextPc)) // Ordinary LSU operations already complete in order.
      is(Kind.exit) {
        assert(registers(17) === 0.U, "Ant unsupported ECALL: pc=0x%x a7=0x%x", pc, registers(17))
        when(registers(17) === 0.U)(finish(registers(10)))
      }
      is(Kind.breakpoint)(assert(false.B, "Ant breakpoint: pc=0x%x", pc))
    }
  }

  val mask = MuxLookup(size, 0.U(p.data.beatBytes.W))(Seq(
    0.U -> 1.U,
    1.U -> 3.U,
    2.U -> 15.U,
    3.U -> 255.U
  ))

  for ((port, shared) <- Seq((io.tls, false), (io.tss, true))) {
    port.request.valid        := state === memory && memoryShared === shared.B && live
    port.request.bits         := 0.U.asTypeOf(port.request.bits)
    port.request.bits.address := memoryAddress
    port.request.bits.size    := size
    port.request.bits.write   := d.kind === Kind.store
    port.request.bits.data    := rhs
    port.request.bits.mask    := Mux(d.kind === Kind.store, mask, 0.U)
    port.response.ready       := (state === memoryWait && memoryShared === shared.B) || (state === complete && cancelled)
  }
  when(io.tls.request.fire || io.tss.request.fire)(state := memoryWait)
  val memoryResponse = Mux(memoryShared, io.tss.response.bits, io.tls.response.bits)
  val memoryValid    = Mux(memoryShared, io.tss.response.valid, io.tls.response.valid)

  val loaded = MuxLookup(d.funct3, memoryResponse.data(63, 0))(Seq(
    0.U -> memoryResponse.data(7, 0).asSInt.pad(64).asUInt,
    1.U -> memoryResponse.data(15, 0).asSInt.pad(64).asUInt,
    2.U -> memoryResponse.data(31, 0).asSInt.pad(64).asUInt,
    4.U -> memoryResponse.data(7, 0).pad(64),
    5.U -> memoryResponse.data(15, 0).pad(64),
    6.U -> memoryResponse.data(31, 0).pad(64)
  ))

  when(state === memoryWait && memoryValid && live) {
    assert(!memoryResponse.error, "Ant local memory access failed: pc=0x%x address=0x%x", pc, memoryAddress)
    when(!memoryResponse.error)(retire(Mux(d.kind === Kind.store, 0.U, d.rd), loaded, nextPc))
  }

  multiplier.io.request.valid                       := state === multiply && live
  multiplier.io.request.bits.a                      := a
  multiplier.io.request.bits.b                      := rhs
  multiplier.io.request.bits.high                   := d.funct3 =/= 0.U
  multiplier.io.request.bits.signedA                := d.funct3 === 1.U || d.funct3 === 2.U
  multiplier.io.request.bits.signedB                := d.funct3 === 1.U
  multiplier.io.response.ready                      := state === multiplyWait || (state === complete && cancelled)
  when(multiplier.io.request.fire) { state := multiplyWait; multiplyPending := true.B }
  when(multiplier.io.response.fire)(multiplyPending := false.B)
  when(state === multiplyWait && multiplier.io.response.fire && live) {
    retire(d.rd, word(multiplier.io.response.bits), nextPc)
  }
  divider.io.clear                                  := false.B
  divider.io.request.valid                          := state === divide && live
  divider.io.request.bits.a                         := a
  divider.io.request.bits.b                         := rhs
  divider.io.request.bits.sew                       := Mux(d.word, 2.U, 3.U)
  divider.io.request.bits.signed                    := !d.funct3(0)
  divider.io.request.bits.remainder                 := d.funct3(1)
  divider.io.response.ready                         := state === divideWait || (state === complete && cancelled)
  when(divider.io.request.fire) { state := divideWait; dividePending := true.B }
  when(divider.io.response.fire)(dividePending      := false.B)
  when(state === divideWait && divider.io.response.fire && live) {
    retire(d.rd, word(divider.io.response.bits), nextPc)
  }

  io.rocc.cmd.valid                     := state === issue && live
  io.rocc.cmd.bits.instruction.raw_inst := instruction
  io.rocc.cmd.bits.instruction.pc       := pc
  io.rocc.cmd.bits.instruction.rs1Data  := Mux(instruction(13), a, 0.U)
  io.rocc.cmd.bits.instruction.rs2Data  := Mux(instruction(12), rhs, 0.U)
  io.rocc.cmd.bits.satp                 := context.satp
  io.rocc.cmd.bits.effectivePrivilege   := context.effectivePrivilege
  io.rocc.cmd.bits.sum                  := context.sum
  io.rocc.cmd.bits.mxr                  := context.mxr
  io.rocc.cmd.bits.pmp                  := context.pmp
  io.rocc.resp.ready                    := false.B
  when(io.rocc.cmd.valid) {
    assert(!instruction(14), "Ant accelerator commands do not write integer registers")
  }
  when(io.rocc.cmd.fire)(retire(0.U, 0.U, nextPc))
  assert(!io.rocc.resp.valid, "Ant received an unsolicited RoCC response")
  assert(!io.rocc.interrupt, "Ant accelerator fault")
  when(io.result.fire)(state            := idle)
  when(io.cancel && state =/= idle && state =/= complete) {
    finish(0.U, true.B)
    cancelled        := true.B
    io.retired.valid := false.B
  }
  // Local execution may have ended while the task still waits for NPU/DMA drain.
  // Once a completion is published, preserve it under response backpressure.
  when(io.cancel && state === complete && !io.result.valid) {
    completion.cancelled := true.B
    completion.value     := 0.U
    cancelled            := true.B
  }
}
