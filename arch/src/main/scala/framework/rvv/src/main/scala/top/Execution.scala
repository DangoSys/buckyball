package framework.rvv

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig

@instantiable
class Execution(val b: GlobalConfig) extends Module {
  val p = b.rvv

  @public
  val io = IO(new Bundle {
    val program        = Flipped(Decoupled(new ProgramWrite))
    val launch         = Flipped(Decoupled(new KernelLaunch))
    val done           = Decoupled(new KernelCompletion)
    val busy           = Output(Bool())
    val ballRequest    = Decoupled(new RvvBallCommand)
    val ballResponse   = Flipped(Decoupled(UInt(64.W)))
    val memoryRequest  = Vec(p.memoryPorts, Decoupled(new VectorMemoryRequest))
    val memoryResponse = Vec(p.memoryPorts, Flipped(Decoupled(new VectorMemoryResponse)))
  })

  val iBuf:      Seq[Instance[IBuf]]       = Seq.fill(2)(Instantiate(new IBuf(b)))
  val registers: Instance[ScalarRF]        = Instantiate(new ScalarRF)
  val scalar:    Instance[ScalarExecution] = Instantiate(new ScalarExecution)
  val floating:  Instance[ScalarFloating]  = Instantiate(new ScalarFloating(p.eLen))
  val vector:    Instance[VectorCore]      = Instantiate(new VectorCore(b))

  val idle :: fetch :: fetchWait :: execute :: vectorWait :: scalarWait :: floatWait :: memorySend :: memoryWait :: ballSend :: ballWait :: completed :: Nil =
    Enum(12)

  val state        = RegInit(idle)
  val activeBuffer = RegInit(false.B)
  val end          = Reg(UInt(32.W))
  val pc           = Reg(UInt(32.W))
  val instruction  = RegInit(0.U(32.W))
  val cycles       = RegInit(0.U(64.W))
  val fault        = RegInit(false.B)
  val cause        = RegInit(0.U(32.W))
  val tval         = RegInit(0.U(64.W))
  val fflags       = RegInit(0.U(5.W))
  val frm          = RegInit(0.U(3.W))
  val vxrm         = RegInit(0.U(2.W))
  val vxsat        = RegInit(false.B)
  val scalarTarget = Reg(UInt(5.W))
  val scalarNextPc = Reg(UInt(32.W))

  val scalarDivide = (instruction(6, 0) === "h33".U || instruction(6, 0) === "h3b".U) && instruction(
    31,
    25
  ) === 1.U && instruction(14, 12) >= 4.U

  val memory         = Reg(new VectorMemoryRequest)
  val memoryFloating = Reg(Bool())
  val memoryUnsigned = Reg(Bool())

  def finish(isFault: Bool, code: UInt, value: UInt): Unit = {
    state := completed
    fault := isFault
    cause := code
    tval  := value
  }

  io.ballRequest.valid            := state === ballSend
  io.ballRequest.bits.funct7      := instruction(31, 25)
  io.ballRequest.bits.rs1         := registers.io.xReadData1
  io.ballRequest.bits.rs2         := registers.io.xReadData2
  io.ballResponse.ready           := state === ballWait
  when(io.ballRequest.fire)(state := ballWait)

  val running = state =/= idle && state =/= completed
  io.busy                  := state =/= idle
  io.launch.ready          := state === idle
  io.done.valid            := state === completed
  io.done.bits.fault       := fault
  io.done.bits.pc          := pc
  io.done.bits.instruction := instruction
  io.done.bits.cycles      := cycles
  io.done.bits.cause       := cause
  io.done.bits.tval        := tval
  io.done.bits.fflags      := fflags
  io.done.bits.vxsat       := vxsat

  for (i <- 0 until 2) {
    iBuf(i).io.upload.valid  := io.program.valid && io.program.bits.buffer === i.U
    iBuf(i).io.upload.bits   := io.program.bits
    iBuf(i).io.uploadEnabled := !io.launch.valid && (!running || activeBuffer =/= i.U)
    iBuf(i).io.fetchEnabled  := state === fetch && activeBuffer === i.U && pc < end
    iBuf(i).io.fetchAddress  := pc
  }
  io.program.ready := Mux(io.program.bits.buffer, iBuf(1).io.upload.ready, iBuf(0).io.upload.ready)
  val imageWords         = Mux(io.launch.bits.iBuffer, iBuf(1).io.wordCount, iBuf(0).io.wordCount)
  val fetchLoaded        = Mux(activeBuffer, iBuf(1).io.loaded, iBuf(0).io.loaded)
  val fetchedInstruction = Mux(activeBuffer, iBuf(1).io.instruction, iBuf(0).io.instruction)

  registers.io.initialize.valid       := io.launch.fire
  registers.io.initialize.bits        := io.launch.bits
  registers.io.xReadAddress1          := instruction(19, 15)
  registers.io.xReadAddress2          := instruction(24, 20)
  registers.io.fReadAddress1          := instruction(19, 15)
  registers.io.fReadAddress2          := instruction(24, 20)
  registers.io.fReadAddress3          := instruction(31, 27)
  registers.io.xWrite.valid           := false.B
  registers.io.xWrite.bits.address    := instruction(11, 7)
  registers.io.xWrite.bits.data       := 0.U
  registers.io.fWrite.valid           := false.B
  registers.io.fWrite.bits.address    := instruction(11, 7)
  registers.io.fWrite.bits.data       := 0.U
  scalar.io.divRequest.valid          := false.B
  scalar.io.divRequest.bits.a         := registers.io.xReadData1
  scalar.io.divRequest.bits.b         := registers.io.xReadData2
  scalar.io.divRequest.bits.sew       := Mux(instruction(6, 0) === "h3b".U, 2.U, 3.U)
  scalar.io.divRequest.bits.signed    := !instruction(12)
  scalar.io.divRequest.bits.remainder := instruction(13)
  scalar.io.divResponse.ready         := state === scalarWait && !io.launch.fire
  scalar.io.clear                     := io.launch.fire
  scalar.io.instruction               := instruction
  scalar.io.pc                        := pc
  scalar.io.source1                   := registers.io.xReadData1
  scalar.io.source2                   := registers.io.xReadData2
  floating.io.instruction             := instruction
  floating.io.source1                 := registers.io.fReadData1
  floating.io.source2                 := registers.io.fReadData2
  floating.io.source3                 := registers.io.fReadData3
  floating.io.xSource                 := registers.io.xReadData1
  floating.io.frm                     := frm
  floating.io.start                   := false.B
  vector.io.initialize                := false.B
  vector.io.roundingMode              := frm
  vector.io.vxrm                      := vxrm
  vector.io.vstartWrite.valid         := false.B
  vector.io.vstartWrite.bits          := 0.U
  vector.io.issue.valid               := false.B
  vector.io.issue.bits.instruction    := instruction
  vector.io.issue.bits.scalar1        := registers.io.xReadData1
  vector.io.issue.bits.scalar2        := registers.io.xReadData2
  vector.io.issue.bits.floating       := registers.io.fReadData1
  vector.io.result.ready              := state === vectorWait

  for (i <- 0 until p.memoryPorts) {
    io.memoryRequest(i) <> vector.io.memoryRequest(i)
    vector.io.memoryResponse(i) <> io.memoryResponse(i)
  }
  when(io.ballResponse.fire) {
    registers.io.xWrite.valid     := instruction(11, 7) =/= 0.U
    registers.io.xWrite.bits.data := io.ballResponse.bits
    pc                            := pc + 4.U
    state                         := fetch
  }
  when(state === scalarWait && scalar.io.divResponse.fire) {
    registers.io.xWrite.valid        := true.B
    registers.io.xWrite.bits.address := scalarTarget
    registers.io.xWrite.bits.data    := Mux(
      instruction(6, 0) === "h3b".U,
      Cat(Fill(32, scalar.io.divResponse.bits(31)), scalar.io.divResponse.bits(31, 0)),
      scalar.io.divResponse.bits
    )
    pc                               := scalarNextPc
    state                            := fetch
  }
  when(state === memorySend || state === memoryWait) {
    io.memoryRequest(0).valid         := state === memorySend
    io.memoryRequest(0).bits          := memory
    vector.io.memoryRequest(0).ready  := false.B
    vector.io.memoryResponse(0).valid := false.B
    io.memoryResponse(0).ready        := state === memoryWait
  }

  val opcode              = instruction(6, 0)
  val funct3              = instruction(14, 12)
  val vectorInstruction   = opcode === "h57".U ||
    (opcode === "h07".U || opcode === "h27".U) && (funct3 === 0.U || funct3 >= 5.U)
  val floatingInstruction = opcode === "h53".U || opcode === "h43".U || opcode === "h47".U ||
    opcode === "h4b".U || opcode === "h4f".U
  val load                = opcode === "h03".U || opcode === "h07".U
  val store               = opcode === "h23".U || opcode === "h27".U
  val csrAddress          = instruction(31, 20)
  val csrKnown            = WireDefault(true.B)
  val csrValue            = WireDefault(0.U(64.W))
  switch(csrAddress) {
    is("h001".U)(csrValue := fflags)
    is("h002".U)(csrValue := frm)
    is("h003".U)(csrValue := Cat(frm, fflags))
    is("h008".U)(csrValue := vector.io.vstart)
    is("h009".U)(csrValue := vxsat)
    is("h00a".U)(csrValue := vxrm)
    is("h00f".U)(csrValue := Cat(vxrm, vxsat))
    is("hc20".U)(csrValue := vector.io.vl)
    is("hc21".U)(csrValue := vector.io.vtype)
    is("hc22".U)(csrValue := (p.vLen / 8).U)
  }
  csrKnown := Seq(0x001, 0x002, 0x003, 0x008, 0x009, 0x00a, 0x00f, 0xc20, 0xc21, 0xc22)
    .map(a => csrAddress === a.U).reduce(_ || _)
  val csrSource           = Mux(funct3(2), instruction(19, 15), registers.io.xReadData1)
  val csrWrite            = funct3(1, 0) === 1.U || instruction(19, 15) =/= 0.U

  val csrNew = MuxLookup(funct3(1, 0), csrValue)(Seq(
    1.U -> csrSource,
    2.U -> (csrValue | csrSource),
    3.U -> (csrValue & ~csrSource)
  ))

  when(running)(cycles                                         := cycles + 1.U)
  when(io.done.fire)(state                                     := idle)
  when(io.launch.fire) {
    activeBuffer := io.launch.bits.iBuffer
    pc           := io.launch.bits.entry
    end          := io.launch.bits.end
    instruction  := 0.U
    cycles       := 0.U
    fflags       := 0.U
    frm          := 0.U
    fault        := false.B
    cause        := 0.U
    tval         := 0.U
    state        := fetch
    when(io.launch.bits.entry(1, 0).orR || io.launch.bits.end(1, 0).orR) {
      finish(true.B, 0.U, io.launch.bits.entry)
    }.elsewhen(io.launch.bits.entry >= io.launch.bits.end || io.launch.bits.end > imageWords * 4.U) {
      finish(true.B, 1.U, io.launch.bits.entry)
    }
  }
  when(state === fetch) {
    when(pc === end)(finish(false.B, 0.U, 0.U))
      .elsewhen(pc(1, 0).orR)(finish(true.B, 0.U, pc))
      .elsewhen(pc >= end || !fetchLoaded)(finish(true.B, 1.U, pc))
      .otherwise(state := fetchWait)
  }
  when(state === fetchWait) { instruction := fetchedInstruction; state := execute }
  when(state === execute) {
    when(opcode === "h7b".U && funct3 === 3.U) {
      state := ballSend
    }.elsewhen(vectorInstruction) {
      vector.io.issue.valid            := true.B
      when(vector.io.issue.fire)(state := vectorWait)
    }.elsewhen(floatingInstruction) {
      when(!floating.io.legal)(finish(true.B, 2.U, instruction))
        .otherwise {
          floating.io.start             := floating.io.ready
          when(floating.io.ready)(state := floatWait)
        }
    }.elsewhen(load || store) {
      val fp        = opcode === "h07".U || opcode === "h27".U
      val size      = Mux(fp, Mux(funct3 === 3.U, 3.U, 2.U), funct3(1, 0))
      val legal     = Mux(
        fp,
        funct3 === 2.U || (p.eLen == 64).B && funct3 === 3.U,
        Mux(store, funct3 <= 3.U, funct3 <= 6.U)
      )
      val immediate = Mux(
        store,
        Cat(Fill(52, instruction(31)), instruction(31, 25), instruction(11, 7)),
        Cat(Fill(52, instruction(31)), instruction(31, 20))
      )
      val address   = registers.io.xReadData1 + immediate
      memory.address     := address
      memory.write       := store
      memory.data        := Mux(fp, registers.io.fReadData2, registers.io.xReadData2)
      memory.size        := size
      memory.mask        := MuxLookup(size, 1.U(8.W))(Seq(0.U -> 1.U, 1.U -> 3.U, 2.U -> 15.U, 3.U -> 255.U))
      memoryFloating     := fp
      memoryUnsigned     := funct3(2)
      when(!legal)(finish(true.B, 2.U, instruction))
        .otherwise(state := memorySend)
    }.elsewhen(opcode === "h73".U) {
      when(funct3 === 0.U) {
        when(instruction === "h00100073".U)(finish(true.B, 3.U, pc))
          .elsewhen(instruction === "h00000073".U)(finish(true.B, 11.U, 0.U))
          .otherwise(finish(true.B, 2.U, instruction))
      }.elsewhen(funct3 === 4.U || !csrKnown || csrWrite && csrAddress(11, 10) === 3.U) {
        finish(true.B, 2.U, instruction)
      }.otherwise {
        registers.io.xWrite.valid     := true.B
        registers.io.xWrite.bits.data := csrValue
        when(csrWrite) {
          switch(csrAddress) {
            is("h001".U)(fflags := csrNew(4, 0))
            is("h002".U)(frm    := csrNew(2, 0))
            is("h003".U) { frm := csrNew(7, 5); fflags := csrNew(4, 0) }
            is("h008".U) { vector.io.vstartWrite.valid := true.B; vector.io.vstartWrite.bits := csrNew }
            is("h009".U)(vxsat  := csrNew(0))
            is("h00a".U)(vxrm   := csrNew(1, 0))
            is("h00f".U) { vxrm := csrNew(2, 1); vxsat := csrNew(0) }
          }
        }
        pc                            := pc + 4.U
        state                         := fetch
      }
    }.elsewhen(!scalar.io.legal)(finish(true.B, 2.U, instruction))
      .elsewhen(scalar.io.nextPc(63, 32).orR)(finish(true.B, 1.U, scalar.io.nextPc))
      .elsewhen(scalar.io.nextPc(1, 0).orR)(finish(true.B, 0.U, scalar.io.nextPc))
      .elsewhen(scalarDivide) {
        scalar.io.divRequest.valid := true.B
        when(scalar.io.divRequest.fire) {
          scalarTarget := scalar.io.destination
          scalarNextPc := scalar.io.nextPc
          state        := scalarWait
        }
      }.otherwise {
        registers.io.xWrite.valid     := scalar.io.write
        registers.io.xWrite.bits.data := scalar.io.result
        pc                            := scalar.io.nextPc
        state                         := fetch
      }
  }
  when(state === memorySend && io.memoryRequest(0).fire)(state := memoryWait)
  when(state === memoryWait && io.memoryResponse(0).fire) {
    when(io.memoryResponse(0).bits.error)(finish(true.B, Mux(memory.write, 7.U, 5.U), memory.address))
      .otherwise {
        when(!memory.write) {
          val data = io.memoryResponse(0).bits.data
          when(memoryFloating) {
            registers.io.fWrite.valid     := true.B
            registers.io.fWrite.bits.data := Mux(memory.size === 2.U, Cat("hffffffff".U(32.W), data(31, 0)), data)
          }.otherwise {
            registers.io.xWrite.valid     := true.B
            registers.io.xWrite.bits.data := MuxLookup(memory.size, data)(Seq(
              0.U -> Cat(Fill(56, !memoryUnsigned && data(7)), data(7, 0)),
              1.U -> Cat(Fill(48, !memoryUnsigned && data(15)), data(15, 0)),
              2.U -> Cat(Fill(32, !memoryUnsigned && data(31)), data(31, 0))
            ))
          }
        }
        pc    := pc + 4.U
        state := fetch
      }
  }
  when((state === floatWait || state === execute && floatingInstruction) && floating.io.valid) {
    when(!floating.io.legal)(finish(true.B, 2.U, instruction))
      .otherwise {
        registers.io.fWrite.valid     := floating.io.floatWrite
        registers.io.fWrite.bits.data := floating.io.result
        registers.io.xWrite.valid     := floating.io.integerWrite
        registers.io.xWrite.bits.data := floating.io.result
        fflags                        := fflags | floating.io.flags
        pc                            := pc + 4.U
        state                         := fetch
      }
  }
  when(vector.io.result.fire) {
    when(vector.io.result.bits.fault)(finish(true.B, vector.io.result.bits.cause, vector.io.result.bits.tval))
      .otherwise {
        registers.io.xWrite.valid     := vector.io.result.bits.scalarWrite
        registers.io.xWrite.bits.data := vector.io.result.bits.scalarData
        registers.io.fWrite.valid     := vector.io.result.bits.floatWrite
        registers.io.fWrite.bits.data := vector.io.result.bits.floatData
        fflags                        := fflags | vector.io.result.bits.flags
        vxsat                         := vxsat || vector.io.result.bits.saturated
        pc                            := pc + 4.U
        state                         := fetch
      }
  }
}
