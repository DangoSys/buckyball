package framework.hart.execution.pipeline

import framework.generic.contract.memory.access.UncachedRequest
import framework.generic.contract.memory.access.UncachedResponse

import framework.generic.contract.memdomain.DmaFaultControl
import framework.generic.contract.memdomain.MvoverISA

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.hart.execution.configs.ExecutionParam
import framework.hart.execution.instruction.InstructionIO
import framework.hart.execution.register_file.RegisterFileIO
import framework.hart.execution.integer.IntegerIO
import framework.hart.execution.muldiv.MulDivIO
import framework.hart.execution.fpu.FPUIO
import framework.hart.execution.privileged.{PrivilegedIO, Trap}
import framework.hart.memory.access.{VirtualPort, VirtualRequest}
import framework.generic.contract.controller.{Commands}
import framework.generic.bus.chi.rnf.CacheAtomic
import framework.generic.contract.frontend.RoCCIO
import framework.generic.contract.hlink.HartInterrupts

class Retirement extends Bundle {
  val pc          = UInt(64.W)
  val instruction = UInt(32.W)
  val destination = UInt(5.W)
  val data        = UInt(64.W)
  val integer     = Bool()
  val floating    = Bool()
}

/** Architectural sequencing and retirement; functional units are connected by Execution. */
@instantiable
class Sequencer(b: ExecutionParam, localRocc: Boolean) extends Module {
  val hart            = b
  val resetVector     = hart.resetVector
  val hartId          = hart.hartId
  val commandsEnabled = hart.commandsEnabled
  val tracking        = hart.tracking

  @public
  val io = IO(new Bundle {
    val interrupts   = Input(new HartInterrupts)
    val controller   = Option.when(!localRocc)(new Commands(b.tracking, b.integer.xLen, b.privileged.permission))
    val rocc         = Option.when(localRocc)(Flipped(new RoCCIO(b.integer.xLen, b.privileged.permission)))
    val workDrained  = Input(Bool())
    // The DMA page fault the directly attached accelerator is stalled on.
    val fault        = Option.when(b.directAccelerator)(Flipped(new DmaFaultControl))
    val retirement   = Valid(new Retirement)
    val fetch        = Flipped(new VirtualPort(b.memoryAccess))
    val memory       = Flipped(new VirtualPort(b.memoryAccess.copy(lineBits = 0)))
    val timeRequest  = Decoupled(new UncachedRequest(b.memoryAccess.memory))
    val timeResponse = Flipped(Decoupled(new UncachedResponse(b.memoryAccess.memory)))
    val instruction  = Flipped(new InstructionIO(b.instruction))
    val registers    = Flipped(new RegisterFileIO(b.registerFile))
    val integer      = Flipped(new IntegerIO(b.integer))
    val multiply     = Flipped(new MulDivIO(b.muldiv))
    val floating     = Flipped(new FPUIO(b.fpu))
    val privileged   = Flipped(new PrivilegedIO(b.privileged))
  })

  val retirement                                                                                                                                                                                                                                                 = io.retirement
  val starting :: fetchLow :: waitLow :: fetchHigh :: waitHigh :: operands :: execute :: sendMemory :: waitMemory :: sendTime :: waitTime :: waitMultiply :: waitFloating :: reserveCommand :: sendCommand :: waitCommand :: fence :: sleeping :: trapped :: Nil =
    Enum(19)
  val state                                                                                                                                                                                                                                                      = RegInit(starting)
  val pc                                                                                                                                                                                                                                                         = Reg(UInt(64.W))
  val encoded                                                                                                                                                                                                                                                    = Reg(UInt(32.W))
  val half                                                                                                                                                                                                                                                       = Reg(UInt(16.W))
  val compressed                                                                                                                                                                                                                                                 = Reg(Bool())

  val memoryRequest = Reg(new VirtualRequest(b.memoryAccess))

  val trap            = Reg(new Trap)
  val fpRegisters     = Reg(Vec(32, UInt(64.W)))
  val commandResponse = RegInit(false.B)

  io.instruction.bits := encoded
  val insn   = io.instruction.expanded
  val opcode = io.instruction.opcode
  val funct3 = io.instruction.funct3
  val funct7 = io.instruction.funct7
  val rs1    = io.instruction.rs1
  val rs2    = io.instruction.rs2
  val rd     = io.instruction.rd
  io.registers.readAddress(0) := rs1
  io.registers.readAddress(1) := rs2
  val a                = Reg(UInt(64.W))
  val operandB         = Reg(UInt(64.W))
  when(state === operands) {
    a        := io.registers.readData(0)
    operandB := io.registers.readData(1)
    state    := execute
  }
  val immediate        = io.instruction.immediate
  val sequentialPc     = pc + Mux(compressed, 2.U, 4.U)
  val writeInteger     = WireDefault(false.B)
  val writeFloating    = WireDefault(false.B)
  val writeData        = WireDefault(0.U(64.W))
  val commit           = WireDefault(false.B)
  val nextPc           = WireDefault(sequentialPc)
  val exception        = WireDefault(false.B)
  val cause            = WireDefault(2.U(64.W))
  val value            = WireDefault(Mux(compressed, encoded(15, 0).pad(64), encoded.pad(64)))
  val legal            = WireDefault(false.B)
  val startMemory      = WireDefault(false.B)
  val store            = opcode === "h23".U || opcode === "h27".U
  val fpMemory         = opcode === "h07".U || opcode === "h27".U
  val atomic           = opcode === "h2f".U
  val atomicOperation  = funct7(6, 2)
  val writesMemory     = store || (atomic && atomicOperation =/= 2.U)
  val effectiveAddress = a + Mux(atomic, 0.U, immediate)
  val memorySize       = Mux(atomic, funct3, funct3(1, 0))
  val memoryData       = Mux(fpMemory, fpRegisters(rs2), operandB)

  val atomicCode = io.instruction.atomicCode

  io.privileged.hartId          := hartId.U(64.W)
  io.privileged.interrupts      := io.interrupts
  io.privileged.retired         := commit
  io.privileged.floating.valid  := writeFloating || (state === waitFloating && io.floating.response.fire && !io.floating.response.bits.illegal)
  io.privileged.floating.bits   := Mux(state === waitFloating, io.floating.response.bits.flags, 0.U)
  io.privileged.dmaFault        := io.fault.map(_.fault).getOrElse(0.U.asTypeOf(io.privileged.dmaFault))
  io.fault.foreach(_.resolve    := io.privileged.dmaResolve)
  io.privileged.trap.valid      := state === trapped
  io.privileged.trap.bits       := trap
  io.privileged.returning.valid := false.B
  io.privileged.returning.bits  := 0.U
  io.privileged.csr.address     := io.instruction.csrAddress
  io.privileged.csr.operation   := funct3(1, 0)
  io.privileged.csr.operand     := Mux(funct3(2), rs1.pad(64), a)
  io.privileged.csr.write       := funct3(1, 0) === 1.U || rs1 =/= 0.U
  io.privileged.csr.commit      := false.B

  io.registers.write.valid                      := writeInteger && commit
  io.registers.write.bits.address               := rd
  io.registers.write.bits.data                  := writeData
  when(writeFloating && commit)(fpRegisters(rd) := writeData)
  retirement.valid                              := commit
  retirement.bits.pc                            := pc
  retirement.bits.instruction                   := encoded
  retirement.bits.destination                   := rd
  retirement.bits.data                          := writeData
  retirement.bits.integer                       := writeInteger && rd =/= 0.U
  retirement.bits.floating                      := writeFloating

  io.integer.in1                       := a
  io.integer.in2                       := Mux(io.instruction.integerImmediate, immediate, operandB)
  io.integer.dw                        := !io.instruction.integerWord
  io.integer.fn                        := io.instruction.integerFunction
  io.multiply.request.valid            := false.B
  io.multiply.request.bits.a           := a
  io.multiply.request.bits.rhs         := operandB
  io.multiply.request.bits.operation   := funct3
  io.multiply.request.bits.word        := opcode === "h3b".U
  io.multiply.response.ready           := state === waitMultiply
  io.floating.request.valid            := false.B
  io.floating.request.bits.instruction := insn
  io.floating.request.bits.integer     := a
  io.floating.request.bits.a           := fpRegisters(rs1)
  io.floating.request.bits.rhs         := fpRegisters(rs2)
  io.floating.request.bits.c           := fpRegisters(funct7(6, 2))
  io.floating.request.bits.rounding    := io.privileged.rounding
  io.floating.response.ready           := state === waitFloating

  val flushTranslation = WireDefault(false.B)
  io.fetch.flushTranslation  := flushTranslation
  io.memory.flushTranslation := flushTranslation
  // The line of the last fetch: its instructions run without another fetch. It is the line of its
  // virtual address in its fetch context; fence, sfence.vma and a CSR write drop it.
  val lineHeld    = RegInit(false.B)
  val lineAddress = Reg(UInt(58.W))
  val lineContext = Reg(UInt(68.W))
  val lineData    = Reg(UInt(512.W))
  val context     = Cat(io.privileged.privilege, io.privileged.sum, io.privileged.mxr, io.privileged.satp)
  def lineHolds(address: UInt): Bool =
    (b.memoryAccess.lineBits == 512).B && lineHeld && lineAddress === address(63, 6) && lineContext === context
  def halfAt(address:    UInt): UInt = (lineData >> Cat(address(5, 1), 0.U(4.W)))(15, 0)
  val lowHeld  = lineHolds(pc)
  val highHeld = lineHolds(pc + 2.U)
  when(state === fetchLow && !io.privileged.interrupt.valid && lowHeld) {
    val low = halfAt(pc)
    when(low(1, 0) =/= 3.U) {
      encoded := low; compressed := true.B; state := operands
    }.elsewhen(highHeld) {
      encoded := Cat(halfAt(pc + 2.U), low); compressed := false.B; state := operands
    }.otherwise {
      half := low; state := fetchHigh
    }
  }
  when(state === fetchHigh && highHeld) {
    encoded := Cat(halfAt(pc + 2.U), half); compressed := false.B; state := operands
  }
  when(flushTranslation || state === fence || io.privileged.csr.commit)(lineHeld := false.B)
  io.fetch.request.valid          :=
    (state === fetchLow && !io.privileged.interrupt.valid && !lowHeld) || (state === fetchHigh && !highHeld)
  io.fetch.request.bits           := 0.U.asTypeOf(io.fetch.request.bits)
  io.fetch.request.bits.vaddr     := Mux(state === fetchHigh, pc + 2.U, pc)
  io.fetch.request.bits.size      := 1.U
  io.fetch.request.bits.execute   := true.B
  io.fetch.request.bits.privilege := io.privileged.privilege
  io.fetch.request.bits.sum       := io.privileged.sum
  io.fetch.request.bits.mxr       := io.privileged.mxr
  io.fetch.request.bits.satpMode  := io.privileged.satp(63, 60)
  io.fetch.request.bits.rootPpn   := io.privileged.satp(43, 0)
  io.fetch.response.ready         := state === waitLow || state === waitHigh
  io.memory.request.valid         := state === sendMemory
  io.memory.request.bits          := memoryRequest
  io.memory.response.ready        := state === waitMemory

  when(state === starting) { pc := resetVector.U(64.W); state := fetchLow }
  when(state === fetchLow && io.privileged.interrupt.valid) {
    trap.pc := pc; trap.cause := io.privileged.interrupt.bits; trap.value := 0.U; state := trapped
  }
  when(io.fetch.request.fire)(state := Mux(state === fetchHigh, waitHigh, waitLow))
  when(io.fetch.response.fire) {
    val response = io.fetch.response.bits
    if (b.memoryAccess.lineBits == 512) {
      when(!response.accessFault && !response.pageFault && !response.misaligned) {
        lineHeld    := true.B
        lineAddress := Mux(state === waitHigh, pc + 2.U, pc)(63, 6)
        lineContext := context
        lineData    := response.line
      }
    }
    when(response.accessFault || response.pageFault || response.misaligned) {
      trap.pc    := pc
      trap.cause := Mux(response.misaligned, 0.U, Mux(response.pageFault, 12.U, 1.U))
      trap.value := Mux(state === waitHigh, pc + 2.U, pc)
      state      := trapped
    }.elsewhen(state === waitHigh) {
      encoded    := Cat(response.data(15, 0), half)
      compressed := false.B
      state      := operands
    }.elsewhen(response.data(1, 0) =/= 3.U) {
      encoded    := response.data(15, 0)
      compressed := true.B
      state      := operands
    }.otherwise {
      half  := response.data(15, 0)
      state := fetchHigh
    }
  }
  // Every trap, on stderr: the report of a program that dies on one, so not
  // a printf (those print only with +disasm).
  val trapReport = Module(new TrapReport)
  trapReport.io.clock               := clock
  trapReport.io.valid               := state === trapped && !reset.asBool
  trapReport.io.pc                  := trap.pc
  trapReport.io.cause               := trap.cause
  trapReport.io.value               := trap.value
  when(state === trapped) {
    pc := io.privileged.redirect.bits; state := fetchLow
  }

  when(state === execute) {
    switch(opcode) {
      is("h37".U) { legal := true.B; writeData := immediate; writeInteger := true.B; commit := true.B }
      is("h17".U) { legal := true.B; writeData := pc + immediate; writeInteger := true.B; commit := true.B }
      is("h6f".U) {
        legal := true.B; nextPc := pc + immediate; writeData := sequentialPc; writeInteger := true.B; commit := true.B
      }
      is("h67".U) {
        legal     := true.B; nextPc             := (a + immediate) & ~1.U(64.W)
        writeData := sequentialPc; writeInteger := true.B; commit := true.B
      }
      is("h63".U) {
        legal := true.B
        val taken = MuxLookup(funct3, false.B)(Seq(
          0.U -> (a === operandB),
          1.U -> (a =/= operandB),
          4.U -> (a.asSInt < operandB.asSInt),
          5.U -> (a.asSInt >= operandB.asSInt),
          6.U -> (a < operandB),
          7.U -> (a >= operandB)
        ))
        nextPc := Mux(taken, pc + immediate, sequentialPc); commit := true.B
      }
      is("h13".U, "h1b".U) {
        legal     := true.B
        writeData := io.integer.out; writeInteger := true.B; commit := true.B
      }
      is("h33".U, "h3b".U) {
        when(funct7 === 1.U) {
          legal                                := true.B
          io.multiply.request.valid            := legal && !io.instruction.illegal
          when(io.multiply.request.fire)(state := waitMultiply)
        }.otherwise {
          legal     := true.B
          writeData := io.integer.out; writeInteger := true.B; commit := true.B
        }
      }
      is("h03".U, "h23".U, "h07".U, "h27".U) {
        legal       := !fpMemory || io.privileged.floatingEnabled
        startMemory := true.B
      }
      is("h2f".U) {
        legal       := true.B
        startMemory := true.B
      }
      is("h43".U, "h47".U, "h4b".U, "h4f".U, "h53".U) {
        legal                                := io.privileged.floatingEnabled
        io.floating.request.valid            := legal && !io.instruction.illegal
        when(io.floating.request.fire)(state := waitFloating)
      }
      is("h0f".U) {
        legal := true.B
        state := fence
      }
      is("h73".U) {
        when(funct3 =/= 0.U) {
          legal := !io.privileged.csr.illegal
          when(io.privileged.csr.memoryRead && legal && !io.instruction.illegal) {
            state := sendTime
          }.otherwise {
            io.privileged.csr.commit := legal && !io.instruction.illegal
            writeData                := io.privileged.csr.data; writeInteger := true.B; commit := true.B
          }
        }.otherwise {
          switch(insn) {
            is("h00000073".U) {
              legal := true.B; exception := true.B; cause := 8.U + io.privileged.privilege; value := 0.U
            }
            is("h00100073".U) { legal := true.B; exception := true.B; cause := 3.U; value := pc }
            is("h30200073".U) {
              legal                         := io.privileged.privilege === 3.U
              io.privileged.returning.valid := legal; io.privileged.returning.bits := 3.U
              nextPc                        := io.privileged.redirect.bits; commit := true.B
            }
            is("h10200073".U) {
              legal                         := io.privileged.privilege >= 1.U && (io.privileged.privilege === 3.U || !io.privileged.tsr)
              io.privileged.returning.valid := legal; io.privileged.returning.bits := 1.U
              nextPc                        := io.privileged.redirect.bits; commit := true.B
            }
            is("h10500073".U) {
              legal  := io.privileged.privilege >= 1.U && (io.privileged.privilege === 3.U || !io.privileged.tw)
              commit := true.B; state := sleeping
            }
          }
          when(funct7 === "h09".U && rd === 0.U) {
            legal            := io.privileged.privilege >= 1.U && (io.privileged.privilege === 3.U || !io.privileged.tvm)
            flushTranslation := legal; commit := true.B
          }
        }
      }
      is("h2b".U) {
        legal             := commandsEnabled.B && !localRocc.B
        when(legal)(state := (if (localRocc) sendCommand else reserveCommand))
      }
      is("h7b".U) {
        legal             := commandsEnabled.B && (hart.directAccelerator.B || (hart.moves.B && funct7 === MvoverISA.Funct.U))
        when(legal)(state := (if (localRocc) sendCommand else reserveCommand))
      }
    }
    when(startMemory && legal && !io.instruction.illegal) {
      memoryRequest           := 0.U.asTypeOf(memoryRequest)
      memoryRequest.vaddr     := effectiveAddress
      memoryRequest.size      := memorySize
      memoryRequest.write     := store
      memoryRequest.signed    := !funct3(2) && !fpMemory
      memoryRequest.data      := memoryData
      memoryRequest.atomic    := Mux(atomic, atomicCode, CacheAtomic.None.U)
      memoryRequest.privilege := io.privileged.dataPrivilege
      memoryRequest.sum       := io.privileged.sum
      memoryRequest.mxr       := io.privileged.mxr
      memoryRequest.satpMode  := io.privileged.satp(63, 60)
      memoryRequest.rootPpn   := io.privileged.satp(43, 0)
      state                   := sendMemory
    }
    when(!legal || io.instruction.illegal) {
      exception := true.B; cause := 2.U
    }
  }

  when(io.memory.request.fire)(state             := waitMemory)
  when(io.memory.response.fire) {
    val response = io.memory.response.bits
    when(response.misaligned || response.pageFault || response.accessFault) {
      exception := true.B
      cause     := Mux(
        response.misaligned,
        Mux(writesMemory, 6.U, 4.U),
        Mux(response.pageFault, Mux(writesMemory, 15.U, 13.U), Mux(writesMemory, 7.U, 5.U))
      )
      value     := memoryRequest.vaddr
    }.otherwise {
      commit        := true.B
      writeInteger  := !store && !fpMemory
      writeFloating := !store && fpMemory
      writeData     := Mux(fpMemory && funct3 === 2.U, Cat(~0.U(32.W), response.data(31, 0)), response.data)
    }
  }
  when(io.multiply.response.fire) { writeInteger := true.B; writeData := io.multiply.response.bits; commit := true.B }
  when(io.floating.response.fire) {
    when(io.floating.response.bits.illegal)(exception := true.B)
      .otherwise {
        writeInteger := io.floating.response.bits.integer; writeFloating := !io.floating.response.bits.integer
        writeData    := io.floating.response.bits.data; commit           := true.B
      }
  }
  when(state === fence && io.workDrained)(commit := true.B)
  // Waits on accelerator work or on a memory request held for it take interrupts precisely: the
  // instruction has had no effect and runs again after the trap. A stalled accelerator DMA
  // raises one.
  val held = state === fence && !io.workDrained || state === sendMemory && !io.memory.request.fire ||
    state === sendCommand && !(if (localRocc) io.rocc.get.cmd.fire else io.controller.get.command.fire)
  when(held && io.privileged.interrupt.valid) {
    trap.pc := pc; trap.cause := io.privileged.interrupt.bits; trap.value := 0.U; state := trapped
  }
  when(state === sleeping && io.privileged.wake)(state := fetchLow)
  when(commit) {
    pc                                                         := nextPc
    when(!(state === execute && insn === "h10500073".U))(state := fetchLow)
  }
  when(exception) {
    commit                   := false.B
    io.privileged.csr.commit := false.B
    trap.pc                  := pc; trap.cause := cause; trap.value := value; state := trapped
  }

  if (localRocc) {
    val rocc = io.rocc.get
    rocc.cmd.valid                     := state === sendCommand
    rocc.cmd.bits.instruction.raw_inst := insn
    rocc.cmd.bits.instruction.pc       := pc
    rocc.cmd.bits.instruction.rs1Data  := a
    rocc.cmd.bits.instruction.rs2Data  := operandB
    rocc.cmd.bits.satp                 := io.privileged.satp
    rocc.cmd.bits.effectivePrivilege   := io.privileged.dataPrivilege
    rocc.cmd.bits.sum                  := io.privileged.sum
    rocc.cmd.bits.mxr                  := io.privileged.mxr
    rocc.cmd.bits.pmp                  := io.privileged.pmp
    when(rocc.cmd.fire) {
      commandResponse := funct3(2)
      when(funct3(2))(state := waitCommand).otherwise(commit := true.B)
    }
    rocc.resp.ready                    := state === waitCommand
    when(rocc.resp.fire) {
      assert(
        commandResponse && rocc.resp.bits.rd === rd,
        "NPU response destination does not match the accepted command"
      )
      writeInteger := true.B; writeData := rocc.resp.bits.data; commit := true.B
    }
  } else {
    val commandsLive = RegInit(VecInit(Seq.fill(tracking.entries)(false.B)))
    val commandTag   = Reg(UInt(tracking.idBits.W))
    val host         = io.controller.get
    val availableTag = PriorityEncoder(~commandsLive.asUInt)
    when(state === reserveCommand && !commandsLive.asUInt.andR && !reset.asBool) {
      commandTag                 := availableTag
      commandsLive(availableTag) := true.B
      state                      := sendCommand
    }
    host.command.valid := state === sendCommand
    host.command.bits.tag                                                := commandTag
    host.command.bits.instruction.raw_inst                               := insn
    host.command.bits.instruction.pc                                     := pc
    host.command.bits.instruction.rs1Data                                := a
    host.command.bits.instruction.rs2Data                                := operandB
    host.command.bits.satp                                               := io.privileged.satp
    host.command.bits.effectivePrivilege                                 := io.privileged.dataPrivilege
    host.command.bits.sum                                                := io.privileged.sum
    host.command.bits.mxr                                                := io.privileged.mxr
    host.command.bits.pmp                                                := io.privileged.pmp
    when(host.command.fire) {
      commandResponse := funct3(2)
      when(funct3(2))(state := waitCommand).otherwise(commit := true.B)
    }
    host.response.ready                                                  := state === waitCommand
    when(host.response.fire) {
      assert(
        commandResponse && host.response.bits.rd === rd,
        "NPU response destination does not match the accepted command"
      )
      writeInteger := true.B; writeData := host.response.bits.data; commit := true.B
    }
    val monitor = Instantiate(new ThreadMonitor)
    monitor.io.clock                                                     := clock
    monitor.io.reset                                                     := reset.asBool
    monitor.io.command                                                   := host.command.fire
    monitor.io.pc                                                        := pc
    monitor.io.funct                                                     := funct7
    monitor.io.rs1                                                       := a
    monitor.io.rs2                                                       := operandB
    monitor.io.response                                                  := host.response.fire
    monitor.io.rd                                                        := host.response.bits.rd
    monitor.io.data                                                      := host.response.bits.data
    monitor.io.state                                                     := state
    host.complete.ready                                                  := true.B
    when(host.complete.fire)(commandsLive(host.complete.bits.tag)        := false.B)
    // A command interrupted before it is sent returns its tag.
    when(held && io.privileged.interrupt.valid)(commandsLive(commandTag) := false.B)
  }
  // Privileged answers an authorization in the cycle after it is presented; a reply held by
  // backpressure retains that decision.
  for ((virtual, index) <- Seq(io.fetch, io.memory).zipWithIndex) {
    val deciding = RegInit(false.B)
    val pending  = RegInit(false.B)
    val allowed  = Reg(Bool())
    io.privileged.authorization(index)                          := virtual.authorizationRequest.bits
    virtual.authorizationRequest.ready                          := !deciding && !pending
    virtual.authorizationResponse.valid                         := deciding || pending
    virtual.authorizationResponse.bits                          := Mux(deciding, io.privileged.permitted(index), allowed)
    when(virtual.authorizationRequest.fire)(deciding            := true.B)
    when(deciding) {
      deciding := false.B; pending := !virtual.authorizationResponse.ready; allowed := io.privileged.permitted(index)
    }
    when(pending && virtual.authorizationResponse.fire)(pending := false.B)
  }

  val timeRequest = io.timeRequest
  timeRequest.valid            := state === sendTime
  timeRequest.bits.addr        := b.mtimeAddress.U
  timeRequest.bits.tag         := 0.U
  timeRequest.bits.size        := 3.U
  timeRequest.bits.write       := false.B
  timeRequest.bits.data        := 0.U
  when(timeRequest.fire)(state := waitTime)
  val timeResponse = io.timeResponse
  timeResponse.ready := state === waitTime
  when(timeResponse.fire) {
    when(timeResponse.bits.error) {
      exception := true.B; cause := 2.U; value := encoded
    }.otherwise {
      writeInteger := true.B; writeData := timeResponse.bits.data; commit := true.B
    }
  }

}

class TrapReport extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val clock = Input(Clock())
    val valid = Input(Bool())
    val pc    = Input(UInt(64.W))
    val cause = Input(UInt(64.W))
    val value = Input(UInt(64.W))
  })

  setInline(
    "TrapReport.v",
    """
      |module TrapReport(
      |  input clock,
      |  input valid,
      |  input [63:0] pc,
      |  input [63:0] cause,
      |  input [63:0] value
      |);
      |`ifndef SYNTHESIS
      |  always @(posedge clock)
      |    if (valid)
      |      $fwrite(32'h80000002, "TRAP pc=0x%x cause=0x%x value=0x%x\n", pc, cause, value);
      |`endif
      |endmodule
      |""".stripMargin
  )
}
