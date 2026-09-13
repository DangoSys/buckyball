package framework.rvv

import chisel3._
import chisel3.util._

case class RvvConfig(vLen: Int = 256, instructionWords: Int = 1024, bankWords: Int = 8192) {
  require(vLen >= 32 && isPow2(vLen))
  require(isPow2(instructionWords) && isPow2(bankWords))
  val elements = vLen / 32
  val vrfWords = 32 * elements
  val vrfBase  = bankWords - vrfWords
  require(vrfBase > 0)
}

/**
 * Sequential kernel engine targeting RV32IF/Zicsr and Zve32f.
 * Host transfers are word addressed; RISC-V PC and load/store addresses are bytes.
 * Vector registers occupy the upper region of the same SRAM bank address space.
 * A launch executes [entry, end); reaching end completes after all writes.
 * Unsupported instructions/configurations and invalid addresses stop with fault.
 */
class KernelEngine(val p: RvvConfig = RvvConfig()) extends Module {

  val io = IO(new Bundle {

    val program = Flipped(Decoupled(new Bundle {
      val address = UInt(32.W)
      val data    = UInt(32.W)
    }))

    val bank = Flipped(Decoupled(new Bundle {
      val address = UInt(32.W)
      val write   = Bool()
      val data    = UInt(32.W)
    }))

    val bankResponse = Decoupled(UInt(32.W))

    val launch = Flipped(Decoupled(new Bundle {
      val entry  = UInt(32.W)
      val end    = UInt(32.W)
      val args   = Vec(8, UInt(32.W))
      val resume = Bool()
    }))

    val done = Decoupled(new Bundle {
      val fault       = Bool()
      val pc          = UInt(32.W)
      val instruction = UInt(32.W)
      val cycles      = UInt(64.W)
      val fflags      = UInt(5.W)
      val vstart      = UInt(32.W)
      val cause       = UInt(32.W)
      val tval        = UInt(32.W)
    })

    val busy = Output(Bool())
  })

  object State extends ChiselEnum {

    val idle, fetch, decode, dispatch, seedWait, readA, waitA, readB, waitB, execute, divide, finish, hostWait,
      hostReply, scalarWait, maskRead, maskWait, readC, waitC, maskWriteRead, maskWriteWait, scalarFloat,
      scalarFloatWait, permuteRead, permuteMaskA, permuteMaskB, permuteIndex, permuteData, memoryStart,
      memoryIndexWait = Value

  }

  import State._

  val state               = RegInit(idle)
  val contextValid        = RegInit(false.B)
  val faultCause          = RegInit(0.U(32.W))
  val faultValue          = RegInit(0.U(32.W))
  val pc                  = RegInit(0.U(32.W))
  val entry               = Reg(UInt(32.W))
  val end                 = Reg(UInt(32.W))
  val instruction         = RegInit(0.U(32.W))
  val fault               = RegInit(false.B)
  val cycles              = RegInit(0.U(64.W))
  val flags               = RegInit(0.U(5.W))
  val x                   = RegInit(VecInit(Seq.fill(32)(0.U(32.W))))
  val f                   = RegInit(VecInit(Seq.fill(32)(0.U(32.W))))
  val vl                  = RegInit(0.U(32.W))
  val vtype               = RegInit("h80000000".U(32.W))
  val vstart              = RegInit(0.U(32.W))
  val vxrm                = RegInit(0.U(2.W))
  val vxsat               = RegInit(false.B)
  val frm                 = RegInit(0.U(3.W))
  val index               = RegInit(0.U(32.W))
  val a                   = Reg(UInt(32.W))
  val b                   = Reg(UInt(32.W))
  val memoryField         = RegInit(0.U(3.W))
  val indexedOffset       = Reg(UInt(32.W))
  val scalarOffset        = Reg(UInt(2.W))
  val scalarFloatOp       = Reg(UInt(6.W))
  val scalarFloatSubop    = Reg(UInt(5.W))
  val scalarFloatRounding = Reg(UInt(3.W))
  val scalarFloatToX      = Reg(Bool())
  val scanCount           = Reg(UInt(32.W))
  val scanSeen            = Reg(Bool())
  val permuteSource       = Reg(UInt(32.W))
  val permuteBit          = Reg(Bool())
  val c                   = Reg(UInt(32.W))
  val maskBit             = Reg(Bool())
  val maskResult          = Reg(Bool())
  val accumulator         = Reg(UInt(32.W))
  val imem                = SyncReadMem(p.instructionWords, UInt(32.W))
  val loaded              = RegInit(VecInit(Seq.fill(p.instructionWords)(false.B)))
  val bank                = SyncReadMem(p.bankWords, Vec(4, UInt(8.W)))
  val bankReadEnable      = WireDefault(false.B)
  val bankReadAddress     = WireDefault(0.U(32.W))
  val bankWriteEnable     = WireDefault(false.B)
  val bankWriteAddress    = WireDefault(0.U(32.W))
  val bankWriteData       = WireDefault(0.U(32.W))
  val bankWriteMask       = WireDefault("b1111".U(4.W))
  val bankData            = bank.read(bankReadAddress(log2Ceil(p.bankWords) - 1, 0), bankReadEnable).asUInt
  when(bankWriteEnable) {
    bank.write(
      bankWriteAddress(log2Ceil(p.bankWords) - 1, 0),
      bankWriteData.asTypeOf(Vec(4, UInt(8.W))),
      bankWriteMask.asBools
    )
  }
  val fetched             = imem.read(pc(log2Ceil(p.instructionWords) + 1, 2), state === fetch && pc < end)

  val op                   = instruction(6, 0)
  val rd                   = instruction(11, 7)
  val funct3               = instruction(14, 12)
  val rs1                  = instruction(19, 15)
  val rs2                  = instruction(24, 20)
  val funct6               = instruction(31, 26)
  val funct7               = instruction(31, 25)
  val immI                 = Cat(Fill(20, instruction(31)), instruction(31, 20))
  val immB                 =
    Cat(Fill(19, instruction(31)), instruction(31), instruction(7), instruction(30, 25), instruction(11, 8), 0.U(1.W))
  val immJ                 =
    Cat(Fill(11, instruction(31)), instruction(31), instruction(19, 12), instruction(20), instruction(30, 21), 0.U(1.W))
  val load                 = op === "h07".U && funct3 =/= 2.U
  val store                = op === "h27".U && funct3 =/= 2.U
  val floatReduction       = op === "h57".U && funct3 === 1.U &&
    (funct6 === 1.U || funct6 === 3.U || funct6 === 5.U || funct6 === 7.U)
  val integerWideReduction = op === "h57".U && funct3 === 0.U && (funct6 === 48.U || funct6 === 49.U)
  val integerReduction     = op === "h57".U && funct3 === 2.U && funct6 <= 7.U || integerWideReduction
  val reduction            = floatReduction || integerReduction
  val moveToFloat          = op === "h57".U && funct3 === 1.U && funct6 === 16.U
  val moveFromFloat        = op === "h57".U && funct3 === 5.U && funct6 === 16.U
  val broadcast            = op === "h57".U && funct3 === 5.U && funct6 === 23.U && instruction(25)
  val floatMerge           = op === "h57".U && funct3 === 5.U && funct6 === 23.U && !instruction(25)
  val sew                  = vtype(4, 3)
  val integer              =
    op === "h57".U && (funct3 === 0.U || funct3 === 2.U || funct3 === 3.U || funct3 === 4.U || funct3 === 6.U)
  val integerMultiply      = funct3 === 2.U || funct3 === 6.U
  val carryOperation       = integer && !integerMultiply && funct6 >= 16.U && funct6 <= 19.U
  val maskOperand          = carryOperation || integer && !integerMultiply && funct6 === 23.U || floatMerge

  val maskDestination = op === "h57".U && Mux(
    integer,
    !integerMultiply && (funct6 >= 24.U && funct6 <= 31.U || funct6 === 17.U || funct6 === 19.U),
    (funct3 === 1.U || funct3 === 5.U) && Seq(24, 25, 27, 28, 29, 31).map(n => funct6 === n.U).reduce(_ || _)
  )

  val floatUnary = op === "h57".U && funct3 === 1.U && (funct6 === 18.U || funct6 === 19.U)

  val needsC = op === "h57".U && Mux(
    integer,
    integerMultiply &&
      (funct6 === 41.U || funct6 === 43.U || funct6 === 45.U || funct6 === 47.U || funct6 >= 60.U),
    (funct3 === 1.U || funct3 === 5.U) && funct6 >= 40.U && funct6 <= 47.U
  )

  val iterativeFloat   = !integer && op === "h57".U &&
    (funct6 === 32.U || funct6 === 33.U || funct6 === 19.U && rs1 === 0.U)
  val integerWide      = integer && integerMultiply && (funct6 >= 48.U && funct6 <= 56.U || funct6 >= 58.U)
  val integerNarrow    = integer && !integerMultiply && funct6 >= 44.U && funct6 <= 47.U
  val integerExtend    = integer && integerMultiply && funct6 === 18.U
  val floatWide        = floatUnary && funct6 === 18.U && (rs1 === 10.U || rs1 === 11.U)
  val floatNarrow      = floatUnary && funct6 === 18.U && (rs1 === 16.U || rs1 === 17.U || rs1 === 22.U || rs1 === 23.U)
  val destinationWidth = Mux(integerWide || floatWide || integerWideReduction, sew + 1.U, sew)

  val sourceAWidth = Mux(
    integerNarrow || floatNarrow || integerWide && funct6 >= 52.U && funct6 <= 55.U,
    sew + 1.U,
    Mux(integerExtend, sew - Mux(rs1(1), 1.U, 2.U), sew)
  )

  val integerToScalar   = op === "h57".U && funct3 === 2.U && funct6 === 16.U && rs1 === 0.U
  val integerFromScalar = op === "h57".U && funct3 === 6.U && funct6 === 16.U && rs2 === 0.U
  val maskLogical       = op === "h57".U && funct3 === 2.U && funct6 >= 24.U && funct6 <= 31.U
  val maskCount         = op === "h57".U && funct3 === 2.U && funct6 === 16.U && (rs1 === 16.U || rs1 === 17.U)
  val maskPrefix        = op === "h57".U && funct3 === 2.U && funct6 === 20.U && rs1 >= 1.U && rs1 <= 3.U
  val iota              = op === "h57".U && funct3 === 2.U && funct6 === 20.U && rs1 === 16.U
  val vectorId          = op === "h57".U && funct3 === 2.U && funct6 === 20.U && rs1 === 17.U && rs2 === 0.U
  val compress          = op === "h57".U && funct3 === 2.U && funct6 === 23.U
  val gather            = op === "h57".U && (funct6 === 12.U && (funct3 === 0.U || funct3 === 3.U || funct3 === 4.U) ||
    funct6 === 14.U && funct3 === 0.U)
  val slide             = op === "h57".U && (funct6 === 14.U || funct6 === 15.U) && funct3 >= 3.U && funct3 <= 6.U
  val slideOne          = slide && (funct3 === 5.U || funct3 === 6.U)
  val wholeMove         = op === "h57".U && funct6 === 39.U && funct3 === 3.U

  val permutation =
    integerToScalar || integerFromScalar || maskLogical || maskCount || maskPrefix || iota || vectorId ||
      compress || gather || slide || wholeMove

  val currentVlmax     =
    Mux(vtype(2), (p.vLen / 8).U(32.W) >> sew >> (8.U - vtype(2, 0)), ((p.vLen / 8).U(32.W) >> sew) << vtype(1, 0))
  val permutationLimit = Mux(wholeMove, ((rs1 +& 1.U) * (p.vLen / 8).U) >> sew, vl)
  val memoryIndexed    = (load || store) && instruction(26)
  val memoryWhole      = (load || store) && instruction(27, 26) === 0.U && rs2 === 8.U
  val memoryMask       = (load || store) && instruction(27, 26) === 0.U && rs2 === 11.U
  val memoryFof        = load && instruction(27, 26) === 0.U && rs2 === 16.U
  val memoryWidth      = MuxLookup(funct3, 2.U(2.W))(Seq(0.U -> 0.U, 5.U -> 1.U))
  val memoryFields     = instruction(31, 29) +& 1.U

  val accessWidth = Mux(
    load || store,
    Mux(memoryMask || memoryWhole && store, 0.U, Mux(memoryIndexed, sew, memoryWidth)),
    destinationWidth
  )

  val memoryEmul      = vtype(2, 0).asSInt.pad(5) + accessWidth.zext - sew.zext
  val memoryGroupSize = Mux(memoryEmul < 0.S, 1.U, 1.U(6.W) << memoryEmul.asUInt(1, 0))
  val memoryRegister  = Mux(memoryWhole || memoryMask, rd, rd + memoryField * memoryGroupSize)
  val memoryLimit     =
    Mux(memoryWhole, (memoryFields * (p.vLen / 8).U) >> accessWidth, Mux(memoryMask, (vl + 7.U) >> 3, vl))

  val memoryAddress = (x(rs1) + Mux(
    memoryWhole || memoryMask,
    index << accessWidth,
    Mux(
      memoryIndexed,
      indexedOffset + (memoryField << accessWidth),
      Mux(
        instruction(27, 26) === 2.U,
        index * x(rs2) + (memoryField << accessWidth),
        (index * memoryFields + memoryField) << accessWidth
      )
    )
  ))(31, 0)

  val permuteReadState = Mux(permutation, permuteRead, Mux(load || store, memoryStart, readA))
  val byteShift        = Cat((index << accessWidth)(1, 0), 0.U(3.W))
  val elementMask      = MuxLookup(accessWidth, "hffffffff".U)(Seq(0.U -> "hff".U, 1.U -> "hffff".U))
  def vectorAddress(reg: UInt, element: UInt, width: UInt = accessWidth): UInt =
    p.vrfBase.U + reg * p.elements.U + ((element << width) >> 2)
  def readWord(address: UInt): Unit = { bankReadEnable := true.B; bankReadAddress := address }

  def writeWord(address: UInt, data: UInt): Unit = {
    bankWriteEnable := true.B; bankWriteAddress := address; bankWriteData := data
  }

  def writeElement(reg: UInt, element: UInt, data: UInt): Unit = {
    val offset = (element << accessWidth)(1, 0)
    writeWord(vectorAddress(reg, element), data << Cat(offset, 0.U(3.W)))
    bankWriteMask := MuxLookup(accessWidth, 15.U(4.W))(Seq(0.U -> 1.U, 1.U -> 3.U)) << offset
  }

  def trap(cause: UInt = 2.U, value: UInt = instruction): Unit = {
    fault      := true.B
    faultCause := cause
    faultValue := value
    pc         := pc
    state      := finish
  }

  def next(vector: Boolean = false): Unit = {
    pc    := pc + 4.U
    state := fetch
    if (vector) { vstart := 0.U }
  }

  def advancePermutation(): Unit =
    when(index === permutationLimit - 1.U)(next(true))
      .otherwise { index := index + 1.U; state := maskRead }

  def writeX(reg: UInt, data: UInt): Unit = when(reg =/= 0.U)(x(reg) := data)

  val fp                = Module(new Float32)
  val scalarFloatActive = state === scalarFloat || state === scalarFloatWait
  fp.io.c            := c
  fp.io.subop        := Mux(scalarFloatActive, scalarFloatSubop, rs1)
  fp.io.a            := Mux(scalarFloatActive, a, Mux(reduction, accumulator, a))
  fp.io.b            := Mux(scalarFloatActive, b, Mux(reduction, a, b))
  fp.io.op           := Mux(scalarFloatActive, scalarFloatOp, Mux(reduction && funct6 === 3.U, 0.U, funct6))
  fp.io.roundingMode := Mux(scalarFloatActive, scalarFloatRounding, frm)
  fp.io.start        := state === scalarFloat || state === execute && !integer && !load && !store && !moveToFloat && !moveFromFloat && !broadcast && !floatMerge

  val intAlu = Module(new Integer32)
  intAlu.io.c             := c
  intAlu.io.carry         := Mux(funct6 === 23.U && instruction(25), true.B, maskBit)
  intAlu.io.vxrm          := vxrm
  intAlu.io.selector      := rs1
  intAlu.io.a             := Mux(integerReduction, accumulator, a)
  intAlu.io.b             := Mux(
    integerReduction,
    Mux(
      integerWideReduction && funct6 === 49.U,
      Mux(sew === 0.U, Cat(Fill(24, a(7)), a(7, 0)), Cat(Fill(16, a(15)), a(15, 0))),
      a
    ),
    b
  )
  intAlu.io.sew           := Mux(integerReduction, destinationWidth, sew)
  intAlu.io.funct6        := Mux(
    integerReduction,
    Mux(integerWideReduction, 0.U, MuxLookup(funct6, funct6)(Seq(1.U -> 9.U, 2.U -> 10.U, 3.U -> 11.U))),
    funct6
  )
  intAlu.io.multiplyClass := (funct3 === 2.U || funct3 === 6.U) && !integerReduction

  io.busy               := state =/= idle
  io.program.ready      := state === idle && !io.bank.valid && !io.launch.valid
  io.bank.ready         := state === idle && !io.launch.valid
  io.launch.ready       := state === idle
  io.bankResponse.valid := state === hostReply
  val hostData = Reg(UInt(32.W))
  io.bankResponse.bits                                                                         := hostData
  io.done.valid                                                                                := state === finish
  io.done.bits.fault                                                                           := fault
  io.done.bits.pc                                                                              := pc
  io.done.bits.instruction                                                                     := instruction
  io.done.bits.cycles                                                                          := cycles
  io.done.bits.fflags                                                                          := flags
  io.done.bits.cause                                                                           := faultCause
  io.done.bits.tval                                                                            := faultValue
  io.done.bits.vstart                                                                          := vstart
  when(state =/= idle && state =/= finish && state =/= hostWait && state =/= hostReply)(cycles := cycles + 1.U)
  when(io.program.fire) {
    assert(io.program.bits.address < p.instructionWords.U, "instruction upload out of range")
    imem.write(io.program.bits.address(log2Ceil(p.instructionWords) - 1, 0), io.program.bits.data)
    loaded(io.program.bits.address(log2Ceil(p.instructionWords) - 1, 0)) := true.B
  }
  when(io.bank.fire) {
    assert(io.bank.bits.address < p.bankWords.U, "bank transfer out of range")
    when(io.bank.bits.write)(writeWord(io.bank.bits.address, io.bank.bits.data))
      .otherwise { readWord(io.bank.bits.address); state := hostWait }
  }
  when(io.launch.fire) {
    when(!io.launch.bits.resume) {
      x.foreach(_ := 0.U)
      f.foreach(_ := 0.U)
      for (i <- 0 until 8) { x(10 + i) := io.launch.bits.args(i) }
      entry        := io.launch.bits.entry
      flags        := 0.U
      vl           := 0.U
      vtype        := "h80000000".U
      vstart       := 0.U
      vxrm         := 0.U
      vxsat        := false.B
      frm          := 0.U
      contextValid := true.B
    }
    pc          := io.launch.bits.entry
    end         := io.launch.bits.end
    cycles      := 0.U
    instruction := 0.U
    fault       := false.B
    faultCause  := 0.U
    faultValue  := 0.U
    state       := fetch
    when(io.launch.bits.entry(1, 0).orR || io.launch.bits.end(1, 0).orR ||
      io.launch.bits.entry >= io.launch.bits.end || io.launch.bits.end > (p.instructionWords * 4).U ||
      io.launch.bits.resume && (!contextValid || io.launch.bits.entry < entry)) {
      trap(Mux(io.launch.bits.entry(1, 0).orR, 0.U, 1.U), io.launch.bits.entry)
      pc := io.launch.bits.entry
    }
  }
  when(scalarFloatActive) {
    when(fp.io.valid) {
      when(scalarFloatToX)(writeX(rd, fp.io.result)).otherwise(f(rd) := fp.io.result)
      flags                                                          := flags | fp.io.flags
      next()
    }.elsewhen(state === scalarFloat && fp.io.ready)(state := scalarFloatWait)
  }
  switch(state) {
    is(hostWait) { hostData := bankData; state := hostReply }
    is(hostReply)(when(io.bankResponse.fire)(state := idle))
    is(finish)(when(io.done.fire)(state := idle))
    is(fetch) {
      when(pc === end)(state := finish)
        .elsewhen(pc < entry || pc > end || pc(1, 0).orR || !loaded(pc(log2Ceil(p.instructionWords) + 1, 2))) {
          trap(Mux(pc(1, 0).orR, 0.U, 1.U), pc)
        }
        .otherwise(state := decode)
    }
    is(decode) {
      // Decode uses the synchronous instruction output; execute fields use its latch.
      instruction := fetched
      state       := dispatch
    }
    is(dispatch) {
      val arithmetic  = (funct3 === 1.U || funct3 === 5.U) && (
        Seq(0, 2, 4, 6, 8, 9, 10, 24, 25, 27, 28, 32, 36, 40, 41, 42, 43, 44, 45, 46, 47).map(n =>
          funct6 === n.U
        ).reduce(_ || _) ||
          funct3 === 5.U && Seq(29, 31, 33, 39).map(n => funct6 === n.U).reduce(_ || _)
      )
      val unaryLegal  = floatUnary && Mux(
        funct6 === 18.U,
        Seq(0, 1, 2, 3, 6, 7, 10, 11, 16, 17, 22, 23).map(n => rs1 === n.U).reduce(_ || _),
        Seq(0, 4, 5, 16).map(n => rs1 === n.U).reduce(_ || _)
      )
      val toF         = moveToFloat && rs1 === 0.U && instruction(25)
      val fromF       = moveFromFloat && rs2 === 0.U && instruction(25)
      val integerForm = integerReduction && (!integerWideReduction || sew < 2.U) || integer && intAlu.io.legal &&
        (funct3 =/= 3.U || Seq(0, 3, 9, 10, 11, 16, 17, 23, 24, 25, 28, 29, 30, 31, 32, 33, 37, 40, 41, 42, 43, 44, 45,
          46, 47)
          .map(n => funct6 === n.U).reduce(_ || _)) &&
        !(funct3 === 0.U && (funct6 === 3.U || funct6 === 30.U || funct6 === 31.U)) &&
        (!carryOperation || !instruction(25) || funct6 === 17.U || funct6 === 19.U) &&
        !(integer && !integerMultiply && funct6 === 23.U && instruction(25) && rs2 =/= 0.U) &&
        !(integerExtend && funct3 =/= 2.U) && !(integerMultiply && funct6 === 62.U && funct3 =/= 6.U)
      val splat       = broadcast && rs2 === 0.U
      next()
      // Vector instructions retire only after their final write.
      when((op === "h57".U && funct3 =/= 7.U) || load || store)(pc := pc)
      switch(op) {
        is("h37".U)(writeX(rd, Cat(instruction(31, 12), 0.U(12.W))))
        is("h17".U)(writeX(rd, pc + Cat(instruction(31, 12), 0.U(12.W))))
        is("h13".U) {
          when(funct3 === 0.U)(writeX(rd, x(rs1) + immI))
            .elsewhen(funct3 === 1.U && funct7 === 0.U)(writeX(rd, x(rs1) << rs2))
            .elsewhen(funct3 === 5.U && funct7 === 0.U)(writeX(rd, x(rs1) >> rs2))
            .elsewhen(funct3 === 5.U && funct7 === 32.U)(writeX(rd, (x(rs1).asSInt >> rs2).asUInt))
            .elsewhen(funct3 === 2.U)(writeX(rd, (x(rs1).asSInt < immI.asSInt).asUInt))
            .elsewhen(funct3 === 3.U)(writeX(rd, (x(rs1) < immI).asUInt))
            .elsewhen(funct3 === 7.U)(writeX(rd, x(rs1) & immI))
            .elsewhen(funct3 === 6.U)(writeX(rd, x(rs1) | immI))
            .elsewhen(funct3 === 4.U)(writeX(rd, x(rs1) ^ immI))
            .otherwise(trap())
        }
        is("h33".U) {
          when(funct7 === 1.U) {
            val signedProduct   = x(rs1).asSInt * x(rs2).asSInt
            val mixedProduct    = x(rs1).asSInt * Cat(0.U(1.W), x(rs2)).asSInt
            val unsignedProduct = x(rs1) * x(rs2)
            val signedOverflow  = x(rs1) === "h80000000".U && x(rs2) === "hffffffff".U
            val signedQuotient  = x(rs1).asSInt / x(rs2).asSInt
            val signedRemainder = x(rs1).asSInt - signedQuotient * x(rs2).asSInt
            val result          = MuxLookup(funct3, 0.U)(Seq(
              0.U -> unsignedProduct(31, 0),
              1.U -> signedProduct.asUInt(63, 32),
              2.U -> mixedProduct.asUInt(63, 32),
              3.U -> unsignedProduct(63, 32),
              4.U -> Mux(x(rs2) === 0.U, "hffffffff".U, Mux(signedOverflow, x(rs1), signedQuotient.asUInt)),
              5.U -> Mux(x(rs2) === 0.U, "hffffffff".U, x(rs1) / x(rs2)),
              6.U -> Mux(x(rs2) === 0.U, x(rs1), Mux(signedOverflow, 0.U, signedRemainder.asUInt)),
              7.U -> Mux(x(rs2) === 0.U, x(rs1), x(rs1) % x(rs2))
            ))
            writeX(rd, result)
          }.elsewhen(funct7 === 0.U || funct7 === 32.U && (funct3 === 0.U || funct3 === 5.U)) {
            val result = MuxLookup(funct3, 0.U)(Seq(
              0.U -> Mux(funct7 === 32.U, x(rs1) - x(rs2), x(rs1) + x(rs2)),
              1.U -> (x(rs1) << x(rs2)(4, 0)),
              2.U -> (x(rs1).asSInt < x(rs2).asSInt).asUInt,
              3.U -> (x(rs1) < x(rs2)).asUInt,
              4.U -> (x(rs1) ^ x(rs2)),
              5.U -> Mux(funct7 === 32.U, (x(rs1).asSInt.pad(33) >> x(rs2)(4, 0)).asUInt(31, 0), x(rs1) >> x(rs2)(4, 0)),
              6.U -> (x(rs1) | x(rs2)),
              7.U -> (x(rs1) & x(rs2))
            ))
            writeX(rd, result)
          }.otherwise(trap())
        }
        is("h63".U) {
          val legal = funct3 === 0.U || funct3 === 1.U || funct3 >= 4.U
          val taken = MuxLookup(funct3, false.B)(Seq(
            0.U -> (x(rs1) === x(rs2)),
            1.U -> (x(rs1) =/= x(rs2)),
            4.U -> (x(rs1).asSInt < x(rs2).asSInt),
            5.U -> (x(rs1).asSInt >= x(rs2).asSInt),
            6.U -> (x(rs1) < x(rs2)),
            7.U -> (x(rs1) >= x(rs2))
          ))
          when(!legal)(trap()).elsewhen(taken) {
            val target = pc + immB
            when(target(1, 0).orR)(trap(0.U, target)).otherwise(pc := target)
          }
        }
        is("h6f".U) {
          val target = pc + immJ
          when(target(1, 0).orR)(trap(0.U, target))
            .otherwise { writeX(rd, pc + 4.U); pc := target }
        }
        is("h67".U) {
          when(funct3 === 0.U) {
            val target = (x(rs1) + immI) & "hfffffffe".U
            when(target(1, 0).orR)(trap(0.U, target))
              .otherwise { writeX(rd, pc + 4.U); pc := target }
          }
            .otherwise(trap())
        }
        is("h03".U) {
          val address   = x(rs1) + immI
          val alignment = (1.U(3.W) << funct3(1, 0)) - 1.U
          when((funct3 <= 2.U || funct3 === 4.U || funct3 === 5.U) &&
            !(address & alignment).orR && (address >> 2) < p.vrfBase.U) {
            readWord(address >> 2)
            scalarOffset := address(1, 0)
            state        := scalarWait
            pc           := pc
          }.otherwise {
            when(!(funct3 <= 2.U || funct3 === 4.U || funct3 === 5.U))(trap())
              .otherwise(trap(Mux((address & alignment).orR, 4.U, 5.U), address))
          }
        }
        is("h23".U) {
          val immediate = Cat(Fill(20, instruction(31)), instruction(31, 25), instruction(11, 7))
          val address   = x(rs1) + immediate
          val alignment = (1.U(3.W) << funct3(1, 0)) - 1.U
          when(funct3 <= 2.U && !(address & alignment).orR && (address >> 2) < p.vrfBase.U) {
            writeWord(address >> 2, x(rs2) << Cat(address(1, 0), 0.U(3.W)))
            bankWriteMask := MuxLookup(funct3, 15.U(4.W))(Seq(0.U -> 1.U, 1.U -> 3.U)) << address(1, 0)
          }.otherwise {
            when(funct3 > 2.U)(trap())
              .otherwise(trap(Mux((address & alignment).orR, 6.U, 7.U), address))
          }
        }
        is("h0f".U) {
          // There are no outstanding accesses in this in-order engine.
          when(funct3 =/= 0.U)(trap())
        }
        is("h73".U) {
          when(instruction === "h00000073".U)(trap(11.U, 0.U))
            .elsewhen(instruction === "h00100073".U)(trap(3.U, pc))
            .otherwise {
              val csr      = instruction(31, 20)
              val source   = Mux(funct3(2), rs1, x(rs1))
              val write    = funct3(1, 0) === 1.U || rs1 =/= 0.U
              val knownCsr = Seq(0x001, 0x002, 0x003, 0x008, 0x009, 0x00a, 0x00f, 0xc20, 0xc21, 0xc22)
                .map(n => csr === n.U).reduce(_ || _)
              val old      = MuxLookup(csr, 0.U(32.W))(Seq(
                "h001".U -> flags,
                "h002".U -> frm,
                "h003".U -> Cat(frm, flags),
                "h008".U -> vstart,
                "h009".U -> vxsat.asUInt,
                "h00a".U -> vxrm,
                "h00f".U -> Cat(vxrm, vxsat),
                "hc20".U -> vl,
                "hc21".U -> vtype,
                "hc22".U -> (p.vLen / 8).U
              ))
              val value    = Mux(funct3(1, 0) === 1.U, source, Mux(funct3(1, 0) === 2.U, old | source, old & ~source))
              when(funct3(1, 0) =/= 0.U && knownCsr && !(csr(11, 10) === 3.U && write)) {
                writeX(rd, old)
                when(write) {
                  switch(csr) {
                    is("h001".U)(flags  := value(4, 0))
                    is("h002".U)(frm    := value(2, 0))
                    is("h003".U) { frm := value(7, 5); flags := value(4, 0) }
                    is("h008".U)(vstart := value(log2Ceil(p.vLen) - 1, 0))
                    is("h009".U)(vxsat  := value(0))
                    is("h00a".U)(vxrm   := value(1, 0))
                    is("h00f".U) { vxrm := value(2, 1); vxsat := value(0) }
                  }
                }
              }.otherwise(trap())
            }
        }
        is("h53".U) {
          when(funct7 === "h78".U && rs2 === 0.U && funct3 === 0.U)(f(rd) := x(rs1))
            .elsewhen(funct7 === "h70".U && rs2 === 0.U && funct3 === 0.U)(writeX(rd, f(rs1)))
            .otherwise {
              val arithmetic     = Seq(0x00, 0x04, 0x08, 0x0c).map(n => funct7 === n.U).reduce(_ || _)
              val sqrt           = funct7 === "h2c".U && rs2 === 0.U
              val convertToInt   = funct7 === "h60".U && rs2 <= 1.U
              val convertToFloat = funct7 === "h68".U && rs2 <= 1.U
              val sign           = funct7 === "h10".U && funct3 <= 2.U
              val minmax         = funct7 === "h14".U && funct3 <= 1.U
              val compare        = funct7 === "h50".U && funct3 <= 2.U
              val classify       = funct7 === "h70".U && rs2 === 0.U && funct3 === 1.U
              val usesRounding   = arithmetic || sqrt || convertToInt || convertToFloat
              val rounding       = Mux(usesRounding, Mux(funct3 === 7.U, frm, funct3), 0.U)
              when((usesRounding || sign || minmax || compare || classify) && rounding <= 4.U) {
                a                   := Mux(convertToFloat, x(rs1), f(rs1))
                b                   := f(rs2)
                scalarFloatOp       := MuxCase(
                  0.U,
                  Seq(
                    (funct7 === 4.U)                 -> 2.U,
                    (funct7 === 8.U)                 -> 36.U,
                    (funct7 === 12.U)                -> 32.U,
                    sqrt                             -> 19.U,
                    (convertToInt || convertToFloat) -> 18.U,
                    sign                             -> (8.U + funct3),
                    minmax                           -> Mux(funct3 === 0.U, 4.U, 6.U),
                    compare                          -> MuxLookup(funct3, 24.U)(Seq(0.U -> 25.U, 1.U -> 27.U)),
                    classify                         -> 19.U
                  )
                )
                scalarFloatSubop    := Mux(classify, 16.U, Mux(convertToInt, 1.U - rs2, Mux(convertToFloat, 3.U - rs2, 0.U)))
                scalarFloatRounding := rounding
                scalarFloatToX      := compare || classify || convertToInt
                state               := scalarFloat
                pc                  := pc
              }.otherwise(trap())
            }
        }
        is("h43".U, "h47".U, "h4b".U, "h4f".U) {
          val rounding = Mux(funct3 === 7.U, frm, funct3)
          when(instruction(26, 25) === 0.U && rounding <= 4.U) {
            a                   := f(rs1); b := f(rs2); c := f(instruction(31, 27))
            scalarFloatOp       := MuxLookup(op, 44.U)(Seq("h47".U -> 46.U, "h4b".U -> 47.U, "h4f".U -> 45.U))
            scalarFloatSubop    := 0.U
            scalarFloatRounding := rounding
            scalarFloatToX      := false.B
            state               := scalarFloat
            pc                  := pc
          }.otherwise(trap())
        }
        is("h57".U) {
          when(funct3 === 7.U) {
            val immediate     = instruction(31, 30) === 3.U
            val register      = funct7 === "h40".U
            val legalEncoding = !instruction(31) || immediate || register
            val requestedType = Mux(register, x(rs2), Mux(immediate, instruction(29, 20), instruction(30, 20)))
            val sew           = requestedType(5, 3)
            val lmul          = requestedType(2, 0)
            val legalType     = !requestedType(31, 8).orR && sew <= 2.U &&
              (lmul <= 3.U || lmul === 6.U && sew === 0.U || lmul === 7.U && sew <= 1.U)
            val base          = (p.vLen / 8).U(32.W) >> sew
            val vlmax         = Mux(lmul(2), base >> (8.U - lmul), base << lmul(1, 0))
            val avl           = Mux(immediate, rs1, Mux(rs1 =/= 0.U, x(rs1), Mux(rd =/= 0.U, "hffffffff".U, vl)))
            val newVl         = Mux(legalType, Mux(avl > vlmax, vlmax, avl), 0.U)
            when(legalEncoding) {
              vtype  := Mux(legalType, requestedType, "h80000000".U)
              vl     := newVl
              vstart := 0.U
              writeX(rd, newVl)
            }.otherwise(trap())
          }.otherwise {
            when(permutation) {
              val groupMask             = Mux(vtype(2), 0.U, (1.U(6.W) << vtype(1, 0)) - 1.U)
              val groupSize             = groupMask + 1.U
              val sourceOverlap         = rd < rs2 +& groupSize && rs2 < rd +& groupSize
              val maskOverlap           = rs1 >= rd && rs1 < rd +& groupSize
              val indexEmul             = vtype(2, 0).asSInt.pad(5) + 1.S - sew.zext
              val indexGroupMask        = Mux(indexEmul < 0.S, 0.U, (1.U(6.W) << indexEmul.asUInt(1, 0)) - 1.U)
              val indexGroupSize        = Mux(funct6 === 14.U, indexGroupMask + 1.U, groupSize)
              val indexOverlap          = rs1 < rd +& groupSize && rd < rs1 +& indexGroupSize
              val vectorGroups          = !(rd & groupMask).orR && (vectorId || iota || !(rs2 & groupMask).orR)
              val scalarMove            = integerToScalar || integerFromScalar
              val maskOnly              = maskLogical || maskCount || maskPrefix
              val gatherIndicesLegal    = funct3 =/= 0.U ||
                Mux(
                  funct6 === 14.U,
                  indexEmul >= -3.S && indexEmul <= 3.S && !(rs1 & indexGroupMask).orR,
                  !(rs1 & groupMask).orR
                )
              val overlapLegal          = (!gather || !sourceOverlap && (funct3 =/= 0.U || !indexOverlap)) &&
                (!slide || funct6 =/= 14.U || !sourceOverlap) &&
                (!compress || !sourceOverlap && !maskOverlap) &&
                (!(maskPrefix || iota) || !(rs2 >= rd && rs2 < rd +& Mux(maskPrefix, 1.U, groupSize)))
              val maskEncodingLegal     = !(scalarMove || maskLogical || compress || wholeMove) || instruction(25)
              val restartLegal          = !(maskCount || maskPrefix || iota || compress) || vstart === 0.U
              val predicateOverlapLegal = (instruction(25) || maskOnly || integerToScalar || rd =/= 0.U) &&
                (!maskPrefix || instruction(25) || rd =/= 0.U)
              val nrLegal               = Seq(0, 1, 3, 7).map(n => rs1 === n.U).reduce(_ || _)
              val wholeLegal            = nrLegal && !(rd & rs1).orR && !(rs2 & rs1).orR && instruction(25)
              when(Mux(
                wholeMove,
                wholeLegal,
                !vtype(31) && (scalarMove || maskOnly || vectorGroups) &&
                  gatherIndicesLegal && overlapLegal && maskEncodingLegal && restartLegal && predicateOverlapLegal &&
                  (!slide || funct3 =/= 5.U || sew === 2.U && frm <= 4.U)
              )) {
                index               := Mux(scalarMove, 0.U, vstart)
                scanCount           := Mux(maskCount && rs1 === 17.U, "hffffffff".U, 0.U)
                scanSeen            := false.B
                when(integerToScalar)(state := permuteRead)
                  .elsewhen(vstart >= permutationLimit) {
                    when(maskCount)(writeX(rd, Mux(rs1 === 17.U, "hffffffff".U, 0.U)))
                    next(true)
                  }.otherwise(state := maskRead)
              }.otherwise(trap())
            }.otherwise {
              val groupChecks          = Seq((rd, destinationWidth), (rs2, sourceAWidth), (rs1, sew)).map { case (reg, width) =>
                val emul = vtype(2, 0).asSInt.pad(5) + width.zext - sew.zext
                val mask = Mux(emul < 0.S, 0.U, (1.U(5.W) << emul.asUInt(1, 0)) - 1.U)
                emul >= -3.S && emul <= 3.S && !(reg & mask).orR
              }
              val groupsLegal          = Mux(
                toF || fromF,
                true.B,
                Mux(
                  reduction,
                  groupChecks(1),
                  (maskDestination || groupChecks(0)) && (splat || groupChecks(1) &&
                    (floatUnary || integerExtend || funct3 === 5.U || funct3 === 3.U || funct3 === 4.U || funct3 === 6.U || groupChecks(
                      2
                    )))
                )
              )
              val destinationEmul      = vtype(2, 0).asSInt.pad(5) + destinationWidth.zext - sew.zext
              val destinationRegisters =
                Mux(maskDestination || destinationEmul < 0.S, 1.U, 1.U(6.W) << destinationEmul.asUInt(1, 0))
              val overlapChecks        = Seq((rs2, sourceAWidth), (rs1, sew)).map { case (source, width) =>
                val sourceEmul      = vtype(2, 0).asSInt.pad(5) + width.zext - sew.zext
                val sourceRegisters = Mux(sourceEmul < 0.S, 1.U, 1.U(6.W) << sourceEmul.asUInt(1, 0))
                val overlap         = rd < (source +& sourceRegisters) && source < (rd +& destinationRegisters)
                !overlap || (!maskDestination && destinationWidth === width) ||
                (maskDestination || destinationWidth < width) && rd === source ||
                !maskDestination && destinationWidth > width && sourceEmul >= 0.S &&
                source === (rd +& destinationRegisters) - sourceRegisters
              }
              val overlapsLegal        = reduction || toF || fromF ||
                (splat || overlapChecks(
                  0
                )) && (floatUnary || integerExtend || funct3 === 3.U || funct3 >= 4.U || overlapChecks(1))
              val maskLegal            = instruction(25) || (rd =/= 0.U || reduction || maskDestination)
              when(!vtype(31) && groupsLegal && overlapsLegal && maskLegal &&
                (integerForm || !integer && (vtype(5, 3) === 2.U && !floatWide && !floatNarrow ||
                  vtype(5, 3) === 1.U && (floatWide || floatNarrow)) && frm <= 4.U &&
                  (arithmetic || reduction || toF || fromF || splat || floatMerge || unaryLegal))) {
                index              := Mux(fromF, 0.U, vstart)
                when(toF)(state := readA)
                  .elsewhen(reduction && vstart =/= 0.U)(trap())
                  .elsewhen(vstart >= vl)(next(true))
                  .elsewhen(reduction) { readWord(vectorAddress(rs1, 0.U)); state := seedWait }
                  .otherwise(state := maskRead)
              }.otherwise(trap())
            }
          }
        }
        is("h07".U, "h27".U) {
          when(funct3 === 2.U) {
            val immediate =
              Mux(op === "h07".U, immI, Cat(Fill(20, instruction(31)), instruction(31, 25), instruction(11, 7)))
            val address   = x(rs1) + immediate
            when(!address(1, 0).orR && (address >> 2) < p.vrfBase.U) {
              when(op === "h07".U) { readWord(address >> 2); state := scalarWait; pc := pc }
                .otherwise(writeWord(address >> 2, f(rs2)))
            }.otherwise {
              trap(Mux(op === "h07".U, Mux(address(1, 0).orR, 4.U, 5.U), Mux(address(1, 0).orR, 6.U, 7.U)), address)
            }
          }.otherwise {
            val widthLegal       = funct3 === 0.U || funct3 === 5.U || funct3 === 6.U
            val groupMask        = memoryGroupSize - 1.U
            val indexEmul        = vtype(2, 0).asSInt.pad(5) + memoryWidth.zext - sew.zext
            val indexGroupSize   = Mux(indexEmul < 0.S, 1.U, 1.U(6.W) << indexEmul.asUInt(1, 0))
            val indexOverlap     = rs2 < rd +& memoryFields * memoryGroupSize && rd < rs2 +& indexGroupSize
            val loadOverlapLegal = !indexOverlap || memoryFields === 1.U &&
              (accessWidth === memoryWidth || accessWidth < memoryWidth && rd === rs2 ||
                accessWidth > memoryWidth && indexEmul >= 0.S && rd +& memoryGroupSize === rs2 +& indexGroupSize)
            val indexLegal       = !memoryIndexed || indexEmul >= -3.S && indexEmul <= 3.S &&
              !(rs2 & (indexGroupSize - 1.U)).orR && Mux(
                load,
                loadOverlapLegal,
                !indexOverlap || accessWidth === memoryWidth
              )
            val wholeLegal       = memoryWhole && instruction(25) && widthLegal && (!store || funct3 === 0.U) &&
              Seq(1, 2, 4, 8).map(n => memoryFields === n.U).reduce(_ || _) && !(rd & (memoryFields - 1.U)).orR
            val maskLegal        = memoryMask && !vtype(31) && instruction(25) && memoryFields === 1.U && funct3 === 0.U
            val regularEncoding  = instruction(27, 26) =/= 0.U || rs2 === 0.U || memoryFof
            val regularLegal     = !memoryWhole && !memoryMask && regularEncoding && !vtype(31) && widthLegal &&
              memoryEmul >= -3.S && memoryEmul <= 3.S &&
              (memoryEmul < 0.S || memoryFields * memoryGroupSize <= 8.U) && !(rd & groupMask).orR &&
              rd +& memoryFields * memoryGroupSize <= 32.U && indexLegal && (!load || instruction(25) || rd =/= 0.U)
            when(!instruction(28) && (wholeLegal || maskLegal || regularLegal)) {
              index       := vstart
              memoryField := 0.U
              when(vstart < memoryLimit)(state := maskRead).otherwise(next(true))
            }.otherwise(trap())
          }
        }
      }
      val known =
        Seq(0x37, 0x17, 0x13, 0x33, 0x63, 0x6f, 0x67, 0x53, 0x57, 0x07, 0x27, 0x03, 0x23, 0x73, 0x0f, 0x43, 0x47, 0x4b,
          0x4f).map(n => op === n.U).reduce(_ || _)
      when(!known)(trap())
    }
    is(seedWait) { accumulator := bankData; state := maskRead }
    is(maskRead) {
      when(instruction(25)) { maskBit := false.B; state := permuteReadState }
        .otherwise { readWord(p.vrfBase.U + (index >> 5)); state := maskWait }
    }
    is(maskWait) {
      maskBit := (bankData >> index(4, 0))(0)
      when(maskOperand || (bankData >> index(4, 0))(0))(state := permuteReadState)
        .otherwise {
          when(index === vl - 1.U) {
            when(reduction)(writeElement(rd, 0.U, accumulator))
            when(maskCount)(writeX(rd, scanCount))
            next(true)
          }.otherwise { index := index + 1.U; state := maskRead }
        }
    }
    is(permuteRead) {
      when(integerFromScalar) { writeElement(rd, 0.U, x(rs1)); next(true) }
        .elsewhen(vectorId) { writeElement(rd, index, index); advancePermutation() }
        .elsewhen(maskLogical || maskCount || maskPrefix || iota || compress) {
          readWord(p.vrfBase.U + Mux(compress, rs1, rs2) * p.elements.U + (index >> 5))
          state := permuteMaskA
        }.elsewhen(gather && funct3 === 0.U) {
          readWord(vectorAddress(rs1, index, Mux(funct6 === 14.U, 1.U, sew)))
          state := permuteIndex
        }.otherwise {
          val offset = Mux(slideOne, 1.U, Mux(funct3 === 3.U, rs1, x(rs1)))
          val source = Mux(
            integerToScalar,
            0.U,
            Mux(wholeMove, index, Mux(gather, offset, Mux(funct6 === 14.U, index - offset, index +& offset)))
          )
          when(slideOne && (funct6 === 14.U && index === 0.U || funct6 === 15.U && index === vl - 1.U)) {
            writeElement(rd, index, Mux(funct3 === 5.U, f(rs1), x(rs1))); advancePermutation()
          }.elsewhen(slide && funct6 === 14.U && index < offset)(advancePermutation())
            .elsewhen((gather || slide) && source >= currentVlmax) {
              writeElement(rd, index, 0.U); advancePermutation()
            }.otherwise {
              permuteSource                                    := source
              readWord(vectorAddress(rs2, source, sew)); state := permuteData
            }
        }
    }
    is(permuteIndex) {
      val width  = Mux(funct6 === 14.U, 1.U, sew)
      val source = (bankData >> Cat((index << width)(1, 0), 0.U(3.W))) &
        MuxLookup(width, "hffffffff".U)(Seq(0.U -> "hff".U, 1.U -> "hffff".U))
      when(source >= currentVlmax) { writeElement(rd, index, 0.U); advancePermutation() }
        .otherwise { permuteSource := source; readWord(vectorAddress(rs2, source, sew)); state := permuteData }
    }
    is(permuteMaskA) {
      val bit = (bankData >> index(4, 0))(0)
      when(maskLogical) {
        permuteBit                                                       := bit
        readWord(p.vrfBase.U + rs1 * p.elements.U + (index >> 5)); state := permuteMaskB
      }.elsewhen(compress) {
        when(bit) { permuteSource := index; readWord(vectorAddress(rs2, index, sew)); state := permuteData }
          .otherwise(advancePermutation())
      }.elsewhen(maskCount) {
        val value = Mux(rs1 === 16.U, scanCount + bit.asUInt, Mux(!scanSeen && bit, index, scanCount))
        scanCount := value
        scanSeen  := scanSeen || bit
        when(index === vl - 1.U) { writeX(rd, value); next(true) }
          .otherwise { index := index + 1.U; state := maskRead }
      }.elsewhen(iota) {
        writeElement(rd, index, scanCount)
        scanCount := scanCount + bit.asUInt
        advancePermutation()
      }.otherwise {
        maskResult := MuxLookup(rs1, !scanSeen)(Seq(1.U -> (!scanSeen && !bit), 2.U -> (!scanSeen && bit)))
        scanSeen   := scanSeen || bit
        state      := maskWriteRead
      }
    }
    is(permuteMaskB) {
      val bit = (bankData >> index(4, 0))(0)
      maskResult := MuxLookup(funct6, false.B)(Seq(
        24.U -> (permuteBit && !bit),
        25.U -> (permuteBit && bit),
        26.U -> (permuteBit || bit),
        27.U -> (permuteBit ^ bit),
        28.U -> (permuteBit || !bit),
        29.U -> (!(permuteBit && bit)),
        30.U -> (!(permuteBit || bit)),
        31.U -> (!(permuteBit ^ bit))
      ))
      state      := maskWriteRead
    }
    is(permuteData) {
      val value = (bankData >> Cat((permuteSource << sew)(1, 0), 0.U(3.W))) & elementMask
      when(integerToScalar) {
        writeX(
          rd,
          MuxLookup(sew, value)(Seq(
            0.U -> Cat(Fill(24, value(7)), value(7, 0)),
            1.U -> Cat(Fill(16, value(15)), value(15, 0))
          ))
        )
        next(true)
      }.otherwise {
        writeElement(rd, Mux(compress, scanCount, index), value)
        when(compress)(scanCount := scanCount + 1.U)
        advancePermutation()
      }
    }
    is(scalarWait) {
      val value = bankData >> Cat(scalarOffset, 0.U(3.W))
      val byte  = Cat(Fill(24, !funct3(2) && value(7)), value(7, 0))
      val half  = Cat(Fill(16, !funct3(2) && value(15)), value(15, 0))
      when(op === "h07".U)(f(rd) := bankData)
        .otherwise(writeX(rd, MuxLookup(funct3(1, 0), value)(Seq(0.U -> byte, 1.U -> half))))
      next()
    }
    is(memoryStart) {
      when(memoryIndexed) {
        readWord(vectorAddress(rs2, index, memoryWidth)); state := memoryIndexWait
      }.otherwise(state := readA)
    }
    is(memoryIndexWait) {
      indexedOffset := (bankData >> Cat((index << memoryWidth)(1, 0), 0.U(3.W))) &
        MuxLookup(memoryWidth, "hffffffff".U)(Seq(0.U -> "hff".U, 1.U -> "hffff".U))
      state         := readA
    }
    is(readA) {
      when(moveFromFloat || broadcast) { b := f(rs1); state := Mux(needsC, readC, execute) }
        .otherwise {
          readWord(Mux(
            load,
            memoryAddress >> 2,
            vectorAddress(
              Mux(store, memoryRegister, rs2),
              Mux(moveToFloat, 0.U, index),
              Mux(load || store, accessWidth, sourceAWidth)
            )
          ))
          state := waitA
          when((load || store) && ((memoryAddress >> 2) >= p.vrfBase.U ||
            (memoryAddress & ((1.U(3.W) << accessWidth) - 1.U)).orR)) {
            when(memoryFof && index =/= 0.U) { vl := index; next(true) }
              .otherwise {
                vstart := index
                val misaligned = (memoryAddress & ((1.U(3.W) << accessWidth) - 1.U)).orR
                trap(Mux(load, Mux(misaligned, 4.U, 5.U), Mux(misaligned, 6.U, 7.U)), memoryAddress)
              }
          }
        }
    }
    is(waitA) {
      val width = Mux(load || store, accessWidth, sourceAWidth)
      val shift = Mux(load, Cat(memoryAddress(1, 0), 0.U(3.W)), Cat((index << width)(1, 0), 0.U(3.W)))
      a                  := (bankData >> shift) & MuxLookup(width, "hffffffff".U)(Seq(0.U -> "hff".U, 1.U -> "hffff".U))
      when(load || store || reduction || moveToFloat || floatUnary || integerExtend)(state := execute)
        .elsewhen(funct3 === 5.U) { b := f(rs1); state := Mux(needsC, readC, execute) }
        .elsewhen(integer && (funct3 === 4.U || funct3 === 6.U)) { b := x(rs1); state := Mux(needsC, readC, execute) }
        .elsewhen(integer && funct3 === 3.U) { b := Cat(Fill(27, rs1(4)), rs1); state := execute }
        .otherwise(state := readB)
    }
    is(readB) { readWord(vectorAddress(rs1, index, sew)); state := waitB }
    is(waitB) {
      b         := (bankData >> Cat((index << sew)(1, 0), 0.U(3.W))) & MuxLookup(sew, "hffffffff".U)(Seq(
        0.U -> "hff".U,
        1.U -> "hffff".U
      )); state := Mux(needsC, readC, execute)
    }
    is(readC) { readWord(vectorAddress(rd, index)); state := waitC }
    is(waitC) { c := (bankData >> byteShift) & elementMask; state := execute }
    is(maskWriteRead) { readWord(p.vrfBase.U + rd * p.elements.U + (index >> 5)); state := maskWriteWait }
    is(maskWriteWait) {
      val bit = 1.U(32.W) << index(4, 0)
      writeWord(p.vrfBase.U + rd * p.elements.U + (index >> 5), Mux(maskResult, bankData | bit, bankData & ~bit))
      when(index === vl - 1.U)(next(true)).otherwise { index := index + 1.U; state := maskRead }
    }
    is(execute) {
      when(iterativeFloat)(when(fp.io.ready)(state := divide))
        .otherwise {
          when(moveToFloat) { f(rd) := a; next(true) }
            .otherwise {
              val result =
                Mux(
                  load || store,
                  a,
                  Mux(
                    moveFromFloat || broadcast,
                    b,
                    Mux(floatMerge, Mux(maskBit, b, a), Mux(integer, intAlu.io.result, fp.io.result))
                  )
                )
              when(maskDestination) {
                maskResult := result(0)
                state      := maskWriteRead
              }.elsewhen(reduction) {
                accumulator := result
                when(index === vl - 1.U)(writeElement(rd, 0.U, result))
              }.elsewhen(store) {
                writeWord(memoryAddress >> 2, result << Cat(memoryAddress(1, 0), 0.U(3.W)))
                bankWriteMask := MuxLookup(accessWidth, 15.U(4.W))(Seq(0.U -> 1.U, 1.U -> 3.U)) << memoryAddress(1, 0)
              }
                .otherwise(writeElement(Mux(load, memoryRegister, rd), index, result))
              when(integer)(vxsat := vxsat || intAlu.io.saturated)
                .elsewhen(!load && !store && !moveFromFloat && !broadcast && !floatMerge)(flags := flags | fp.io.flags)
              when(!maskDestination) {
                when((load || store) && !memoryWhole && !memoryMask && memoryField =/= memoryFields - 1.U) {
                  memoryField := memoryField + 1.U; state := readA
                }.otherwise {
                  memoryField := 0.U
                  when(index === Mux(load || store, memoryLimit, vl) - 1.U || moveFromFloat)(next(true))
                    .otherwise { index := index + 1.U; state := maskRead }
                }
              }
            }
        }
    }
    is(divide) {
      when(fp.io.valid) {
        writeWord(vectorAddress(rd, index), fp.io.result)
        flags := flags | fp.io.flags
        when(index === vl - 1.U)(next(true)).otherwise { index := index + 1.U; state := maskRead }
      }
    }
  }
}

object EmitKernelEngine extends App {
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new KernelEngine(),
    args,
    firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info")
  )
}
