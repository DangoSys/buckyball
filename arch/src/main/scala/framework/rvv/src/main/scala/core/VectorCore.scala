package framework.rvv

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig

@instantiable
class VectorCore(val b: GlobalConfig) extends Module {
  val p = b.rvv

  @public
  val io = IO(new Bundle {
    val initialize     = Input(Bool())
    val roundingMode   = Input(UInt(3.W))
    val vxrm           = Input(UInt(2.W))
    val vstartWrite    = Flipped(Valid(UInt(64.W)))
    val vl             = Output(UInt(32.W))
    val vtype          = Output(UInt(64.W))
    val vstart         = Output(UInt(64.W))
    val issue          = Flipped(Decoupled(new VectorIssue))
    val result         = Decoupled(new VectorResult)
    val memoryRequest  = Vec(p.memoryPorts, Decoupled(new VectorMemoryRequest))
    val memoryResponse =
      Vec(p.memoryPorts, Flipped(Decoupled(new VectorMemoryResponse)))
    val busy           = Output(Bool())
  })

  val idle :: snapshot :: capture :: execute :: commit :: Nil = Enum(5)
  val state                                                   = RegInit(idle)
  val resultValid                                             = RegInit(false.B)
  val result                                                  = RegInit(0.U.asTypeOf(new VectorResult))
  val instruction                                             = Reg(UInt(32.W))
  val scalar1                                                 = Reg(UInt(64.W))
  val scalar2                                                 = Reg(UInt(64.W))
  val floatScalar                                             = Reg(UInt(64.W))
  val vl                                                      = RegInit(0.U(32.W))
  val vtype                                                   = RegInit("h8000000000000000".U(64.W))
  val vstart                                                  = RegInit(0.U(64.W))
  val base                                                    = Reg(UInt(32.W))
  val readPortCount                                           = math.max(p.laneNumber, p.memoryPorts)
  val wordsPerRegister                                        = p.wordsPerRegister
  val maskWords                                               = Reg(Vec(readPortCount, UInt(p.wordBits.W)))
  val operandWords                                            = Reg(Vec(readPortCount, UInt(p.wordBits.W)))
  val sourceWords                                             = Reg(Vec(readPortCount, UInt(p.wordBits.W)))
  val destinationWords                                        = Reg(Vec(readPortCount, UInt(p.wordBits.W)))
  val readPhase                                               = RegInit(0.U(3.W))
  val readSent                                                = RegInit(false.B)
  val snapshotSent                                            = RegInit(false.B)
  val firstBatch                                              = RegInit(true.B)
  val writeSent                                               = RegInit(false.B)
  val resumeState                                             = Reg(UInt(3.W))
  val finishRequested                                         = WireDefault(false.B)
  val accumulator                                             = Reg(UInt(p.wordBits.W))
  val count                                                   = Reg(UInt(32.W))
  val offered                                                 = RegInit(VecInit(Seq.fill(p.memoryPorts)(false.B)))
  val issued                                                  = RegInit(VecInit(Seq.fill(p.memoryPorts)(false.B)))
  val received                                                = RegInit(VecInit(Seq.fill(p.memoryPorts)(false.B)))
  val memoryData                                              = Reg(Vec(p.memoryPorts, UInt(p.wordBits.W)))
  val memoryFailed                                            = RegInit(VecInit(Seq.fill(p.memoryPorts)(false.B)))
  val divBatchValid                                           = RegInit(false.B)
  val divActive                                               = RegInit(VecInit(Seq.fill(p.laneNumber)(false.B)))
  val divStarted                                              = RegInit(VecInit(Seq.fill(p.laneNumber)(false.B)))
  val divDone                                                 = RegInit(VecInit(Seq.fill(p.laneNumber)(false.B)))
  val divData                                                 = Reg(Vec(p.laneNumber, UInt(p.wordBits.W)))
  val fpStarted                                               = RegInit(VecInit(Seq.fill(p.laneNumber)(false.B)))
  val fpDone                                                  = RegInit(VecInit(Seq.fill(p.laneNumber)(false.B)))
  val fpData                                                  = Reg(Vec(p.laneNumber, UInt(p.wordBits.W)))
  val fpFlags                                                 = Reg(Vec(p.laneNumber, UInt(5.W)))
  val fpLegal                                                 = Reg(Vec(p.laneNumber, Bool()))

  val opcode        = instruction(6, 0)
  val rd            = instruction(11, 7)
  val kind          = instruction(14, 12)
  val rs1           = instruction(19, 15)
  val rs2           = instruction(24, 20)
  val vm            = instruction(25)
  val op            = instruction(31, 26)
  val sew           = vtype(4, 3)
  val lmul          = Cat(vtype(2), vtype(2, 0)).asSInt
  val configure     = opcode === "h57".U && kind === 7.U
  val load          = opcode === "h07".U
  val store         = opcode === "h27".U
  val memory        = load || store
  val fp            = opcode === "h57".U && (kind === 1.U || kind === 5.U)
  val integer       =
    opcode === "h57".U && (kind === 0.U || kind === 2.U || kind === 3.U || kind === 4.U || kind === 6.U)
  val divide        = integer && (kind === 2.U || kind === 6.U) && op >= 32.U && op <= 35.U
  val multiply      = kind === 2.U || kind === 6.U
  val vectorOperand = kind === 0.U || kind === 1.U || kind === 2.U
  val maskLogical   = integer && kind === 2.U && op >= 24.U && op <= 31.U

  val compare =
    (integer && !multiply && (op >= 24.U && op <= 31.U || op === 17.U || op === 19.U)) ||
      (fp && Seq(24, 25, 27, 28, 29, 31).map(n => op === n.U).reduce(_ || _))

  val carryOperation = integer && !multiply && op >= 16.U && op <= 19.U
  val carryMask      = carryOperation && op(0)
  val merge          = op === 23.U && (fp || integer && !multiply)
  val scalarMoveOut  =
    op === 16.U && rs1 === 0.U && vm && (kind === 2.U || kind === 1.U)
  val scalarMoveIn   =
    op === 16.U && rs2 === 0.U && vm && (kind === 6.U || kind === 5.U)
  val population     = integer && kind === 2.U && op === 16.U && rs1 === 16.U
  val first          = integer && kind === 2.U && op === 16.U && rs1 === 17.U
  val identity       =
    integer && kind === 2.U && op === 20.U && rs1 === 17.U && rs2 === 0.U
  val intReduction   = integer && kind === 2.U && op <= 7.U
  val fpReduction    =
    fp && kind === 1.U && (op === 1.U || op === 3.U || op === 5.U || op === 7.U)
  val reduction      = intReduction || fpReduction
  val wholeMove      = integer && kind === 3.U && op === 39.U
  val memMode        = instruction(27, 26)

  val memWidth = MuxLookup(kind, 0.U(2.W))(
    Seq(0.U -> 0.U, 5.U -> 1.U, 6.U -> 2.U, 7.U -> 3.U)
  )

  val wholeMemory = memory && memMode === 0.U && rs2 === 8.U
  val indexed     = memMode(0)
  val memSew      = Mux(indexed && !wholeMemory, sew, memWidth)
  val intWide     = integer && multiply && op >= 48.U
  val intNarrow   = integer && !multiply && op >= 44.U && op <= 47.U
  val floatWide   = fp && op === 18.U && rs1 >= 8.U && rs1 <= 15.U
  val floatNarrow = fp && op === 18.U && rs1 >= 16.U && rs1 <= 23.U
  val extension   = integer && multiply && op === 18.U

  val sourceSew = Mux(
    memory,
    Mux(indexed, memWidth, memSew),
    Mux(
      intNarrow || floatNarrow || intWide && op >= 52.U && op <= 55.U,
      sew +& 1.U,
      Mux(
        extension,
        sew - Mux(rs1(2, 1) === 1.U, 3.U, Mux(rs1(2, 1) === 2.U, 2.U, 1.U)),
        sew
      )
    )
  )

  val destinationSew =
    Mux(memory, memSew, Mux(intWide || floatWide, sew +& 1.U, sew))

  val limit = Mux(
    wholeMemory,
    ((instruction(31, 29) +& 1.U) * p.vLen.U) >> (memWidth +& 3.U),
    Mux(wholeMove, ((rs1 +& 1.U) * p.vLen.U) >> (sew +& 3.U), vl)
  )

  def element(word: UInt, index: UInt, width: UInt): UInt = {
    val shift = (index << (width.pad(3) + 3.U))(p.wordOffsetBits - 1, 0)
    val mask  = MuxLookup(width, Fill(p.wordBits, 1.U(1.W)))(
      Seq(0.U -> 255.U, 1.U -> 65535.U, 2.U -> "hffffffff".U)
    )
    ((word >> shift) & mask)(p.eLen - 1, 0)
  }

  def maskElement(word: UInt, index: UInt): Bool = (word >> index(p.wordOffsetBits - 1, 0))(0)

  def wordAddress(
    register: UInt,
    index:    UInt,
    width:    UInt,
    maskBit:  Bool = false.B
  ): UInt = {
    val shift   = Mux(maskBit, index, index << (width.pad(3) + 3.U))
    val address = Mux(maskBit, register, register + (shift >> log2Ceil(p.vLen)))(4, 0)
    val within  = shift.pad(log2Ceil(p.vLen))(log2Ceil(p.vLen) - 1, 0)
    address * wordsPerRegister.U + (within >> p.wordOffsetBits)
  }

  def groupLegal(register: UInt, width: UInt): Bool = {
    val exponent  = lmul.pad(6) + width.zext - sew.zext
    val registers = 1.U(6.W) << Mux(exponent > 0.S, exponent.asUInt, 0.U)
    exponent >= -3.S && exponent <= 3.S && (register & (registers - 1.U)) === 0.U && register +& registers <= 32.U
  }

  def overlapLegal(
    destination:      UInt,
    destinationWidth: UInt,
    operand:          UInt,
    operandWidth:     UInt
  ): Bool = {
    val destinationExponent = lmul.pad(6) + destinationWidth.zext - sew.zext
    val operandExponent     = lmul.pad(6) + operandWidth.zext - sew.zext
    val destinationCount    = 1
      .U(6.W) << Mux(destinationExponent > 0.S, destinationExponent.asUInt, 0.U)
    val operandCount        =
      1.U(6.W) << Mux(operandExponent > 0.S, operandExponent.asUInt, 0.U)
    val overlap             =
      destination < operand +& operandCount && operand < destination +& destinationCount
    !overlap || destinationWidth === operandWidth ||
    (destinationWidth < operandWidth && destination === operand) ||
    (destinationWidth > operandWidth && operandExponent >= 0.S && destination +& destinationCount === operand +& operandCount)
  }

  def maximum(width: UInt, multiplier: UInt): UInt = {
    val elements = p.vLen.U(32.W) >> (width.pad(3) + 3.U)
    Mux(multiplier(2), elements >> (8.U - multiplier), elements << multiplier)
  }

  val vlmax = maximum(sew, vtype(2, 0))
  val registers: Instance[Operands] = Instantiate(new Operands(b))
  registers.io.initialize := io.initialize
  val writes        = Wire(chiselTypeOf(registers.io.write.bits))
  val pendingWrites = Reg(chiselTypeOf(registers.io.write.bits))
  for (lane <- 0 until p.laneNumber) {
    writes(lane) := 0.U.asTypeOf(writes(lane))
  }
  registers.io.write.valid := state === commit && !writeSent && !io.initialize
  registers.io.write.bits                 := pendingWrites
  registers.io.writeDone.ready            := state === commit && writeSent && !io.initialize
  when(registers.io.write.fire)(writeSent := true.B)
  when(registers.io.writeDone.fire) {
    writeSent := false.B
    state     := resumeState
  }

  def writeElement(
    lane:    Int,
    index:   UInt,
    width:   UInt,
    data:    UInt,
    maskBit: Bool = false.B
  ): Unit = {
    val bitOffset = Mux(maskBit, index, index << (width.pad(3) + 3.U))
    val within    = bitOffset.pad(log2Ceil(p.vLen))(log2Ceil(p.vLen) - 1, 0)
    val mask      = Mux(
      maskBit,
      1.U(p.wordBits.W),
      MuxLookup(width, Fill(p.wordBits, 1.U(1.W)))(
        Seq(0.U -> 255.U, 1.U -> 65535.U, 2.U -> "hffffffff".U)
      )
    )
    writes(lane).valid := true.B
    writes(lane).address := wordAddress(rd, index, width, maskBit)
    writes(lane).mask    := (mask << within(p.wordOffsetBits - 1, 0))(p.wordBits - 1, 0)
    writes(lane).data    := (data << within(p.wordOffsetBits - 1, 0))(p.wordBits - 1, 0)
  }

  def finish(
    fault: Bool = false.B,
    cause: UInt = 0.U,
    value: UInt = 0.U
  ): Unit = {
    finishRequested     := true.B
    state               := idle
    resultValid         := true.B
    result.fault        := fault
    result.cause        := cause
    result.tval         := value
    when(!fault)(vstart := 0.U)
  }

  def nextBatch(step: UInt): Unit = {
    when(base + step >= limit)(finish()).otherwise {
      base      := base + step
      state     := capture
      readPhase := 0.U
    }
    offered.foreach(_      := false.B)
    issued.foreach(_       := false.B)
    received.foreach(_     := false.B)
    memoryFailed.foreach(_ := false.B)
    divBatchValid          := false.B
    divStarted.foreach(_   := false.B)
    divDone.foreach(_      := false.B)
    fpStarted.foreach(_    := false.B)
    fpDone.foreach(_       := false.B)
  }

  io.vl                             := vl
  io.vtype                          := vtype
  io.vstart                         := vstart
  io.busy                           := state =/= idle || resultValid || registers.io.busy
  io.issue.ready                    := state === idle && !resultValid && !registers.io.busy && !io.initialize
  io.result.valid                   := resultValid && state === idle && !registers.io.busy
  io.result.bits                    := result
  when(io.result.fire)(resultValid  := false.B)
  when(io.vstartWrite.valid)(vstart := io.vstartWrite.bits)
  when(io.issue.fire) {
    instruction            := io.issue.bits.instruction
    scalar1                := io.issue.bits.scalar1
    scalar2                := io.issue.bits.scalar2
    floatScalar            := io.issue.bits.floating
    state                  := snapshot
    readPhase              := 0.U
    readSent               := false.B
    snapshotSent           := false.B
    firstBatch             := true.B
    writeSent              := false.B
    result                 := 0.U.asTypeOf(new VectorResult)
    base                   := vstart
    count                  := 0.U
    offered.foreach(_      := false.B)
    issued.foreach(_       := false.B)
    received.foreach(_     := false.B)
    memoryFailed.foreach(_ := false.B)
    divBatchValid          := false.B
    divStarted.foreach(_   := false.B)
    divDone.foreach(_      := false.B)
    fpStarted.foreach(_    := false.B)
    fpDone.foreach(_       := false.B)
  }

  val instructionLegal = Wire(Bool())
  val alu:  Seq[Instance[IALU]] = Seq.fill(p.laneNumber)(Instantiate(new IALU(p.eLen)))
  val falu: Seq[Instance[FALU]] = Seq.fill(p.laneNumber)(Instantiate(new FALU(p.eLen)))
  val laneA        = Wire(Vec(p.laneNumber, UInt(p.wordBits.W)))
  val laneB        = Wire(Vec(p.laneNumber, UInt(p.wordBits.W)))
  val laneSelected = Wire(Vec(p.laneNumber, Bool()))
  val laneValue    = Wire(Vec(p.laneNumber, UInt(p.wordBits.W)))
  val laneFlags    = Wire(Vec(p.laneNumber, UInt(5.W)))
  val laneLegal    = Wire(Vec(p.laneNumber, Bool()))
  val fpComplete   = Wire(Vec(p.laneNumber, Bool()))
  val gather       =
    integer && !multiply && (op === 12.U || op === 14.U && kind === 0.U)
  val slide        =
    integer && !multiply && (op === 14.U || op === 15.U) && kind =/= 0.U
  val slideOne     = integer && multiply && (op === 14.U || op === 15.U)

  val operandReads = (0 until readPortCount).map { port =>
    val index          = base + port.U
    val scalar         = Mux(kind === 3.U, Cat(Fill(59, rs1(4)), rs1), scalar1)
    val floatingScalar = Mux(sew === 2.U && !floatScalar(63, 32).andR, "h7fc00000".U, floatScalar)
    val b              =
      Mux(
        vectorOperand,
        element(operandWords(port), index, Mux(gather && op === 14.U, 1.U, sew)),
        Mux(fp, floatingScalar, scalar)
      )
    val slideOffset    = Mux(slideOne, 1.U, Mux(kind === 3.U, rs1, scalar1))
    val sourceIndex    = Mux(gather && !slide, b, Mux(op === 14.U, index - slideOffset, index + slideOffset))
    val permuteRead    = gather || slide || slideOne
    // element already keeps only the low group bit offset; upper index bits cannot affect its read.
    val a              = element(
      sourceWords(port),
      Mux(scalarMoveOut, 0.U, Mux(permuteRead, sourceIndex(31, 0), index)),
      Mux(permuteRead, sew, sourceSew)
    )
    val c              = element(destinationWords(port), index, Mux(fp, sew, destinationSew))
    (a, b, c, sourceIndex)
  }

  val permuteRead = gather || slide || slideOne
  registers.io.snapshot.valid                   := state === snapshot && permuteRead && !snapshotSent && !io.initialize
  registers.io.snapshot.bits                    := rs2
  registers.io.snapshotDone.ready               := state === snapshot && snapshotSent && !io.initialize
  when(registers.io.snapshot.fire)(snapshotSent := true.B)
  when(state === snapshot && !permuteRead || registers.io.snapshotDone.fire) {
    state     := Mux(configure, execute, capture)
    readPhase := 0.U
    readSent  := false.B
  }

  registers.io.read.valid       := state === capture && !readSent && !io.initialize
  registers.io.readResult.ready := state === capture && readSent && !io.initialize
  for (port <- 0 until readPortCount) {
    val index       = base + port.U
    val maskSource  = maskLogical || population || first
    val sourceIndex = Mux(scalarMoveOut, 0.U, Mux(permuteRead, operandReads(port)._4(31, 0), index))
    val sourceWidth = Mux(scalarMoveOut || permuteRead, sew, sourceSew)
    val shift       = sourceIndex << (sourceWidth.pad(3) + 3.U)
    registers.io.read.bits(port).address  := MuxLookup(readPhase, wordAddress(rs1, 0.U, sew))(
      Seq(
        0.U -> wordAddress(0.U, index, sew, true.B),
        1.U -> wordAddress(rs1, index, Mux(gather && op === 14.U, 1.U, sew), maskLogical),
        2.U -> Mux(permuteRead, shift >> p.wordOffsetBits, wordAddress(rs2, sourceIndex, sourceWidth, maskSource)),
        3.U -> wordAddress(rd, index, Mux(fp, sew, destinationSew))
      )
    )
    registers.io.read.bits(port).snapshot := readPhase === 2.U && permuteRead
  }
  when(registers.io.read.fire)(readSent := true.B)
  when(registers.io.readResult.fire) {
    readSent              := false.B
    switch(readPhase) {
      is(0.U)(maskWords        := registers.io.readResult.bits)
      is(1.U)(operandWords     := registers.io.readResult.bits)
      is(2.U)(sourceWords      := registers.io.readResult.bits)
      is(3.U)(destinationWords := registers.io.readResult.bits)
      is(4.U) {
        accumulator := registers.io.readResult.bits(0) & MuxLookup(sew, Fill(p.eLen, 1.U(1.W)))(
          Seq(0.U -> 255.U, 1.U -> 65535.U, 2.U -> "hffffffff".U)
        )
        firstBatch  := false.B
      }
    }
    when(readPhase === 4.U || readPhase === 3.U && !firstBatch) {
      state := execute
    }.otherwise(readPhase := readPhase + 1.U)
  }

  for (lane <- 0 until p.laneNumber) {
    val index          = base + lane.U
    val selected       = maskElement(maskWords(lane), index)
    val scalar         = Mux(
      kind === 3.U,
      Cat(Fill(59, rs1(4)), rs1),
      scalar1
    )
    val floatingScalar =
      Mux(sew === 2.U && !floatScalar(63, 32).andR, "h7fc00000".U, floatScalar)
    val b              = operandReads(lane)._2
    val slideOffset    = Mux(slideOne, 1.U, Mux(kind === 3.U, rs1, scalar1))
    val up             = op === 14.U
    val sourceIndex    = operandReads(lane)._4
    val a              = operandReads(lane)._1
    laneA(lane)                                         := a
    laneB(lane)                                         := b
    laneSelected(
      lane
    )                                                   := index < limit && (wholeMove || vm || selected || merge || carryOperation) &&
      (!slide || !up || index >= slideOffset)
    alu(lane).io.divRequest.valid                       := state === execute && instructionLegal && divide &&
      divBatchValid && divActive(lane) && !divStarted(lane) && !io.initialize
    alu(lane).io.divRequest.bits.a                      := a
    alu(lane).io.divRequest.bits.b                      := b
    alu(lane).io.divRequest.bits.sew                    := sew
    alu(lane).io.divRequest.bits.signed                 := op(0)
    alu(lane).io.divRequest.bits.remainder              := op(1)
    alu(lane).io.divResponse.ready                      := state === execute && divide && divBatchValid &&
      divStarted(lane) && !divDone(lane) && !io.initialize
    alu(lane).io.clear                                  := io.initialize
    when(alu(lane).io.divRequest.fire)(divStarted(lane) := true.B)
    when(alu(lane).io.divResponse.fire) {
      divData(lane) := alu(lane).io.divResponse.bits
      divDone(lane) := true.B
    }
    alu(lane).io.a                                      := a
    alu(lane).io.b                                      := b
    alu(lane).io.c                                      := operandReads(lane)._3
    alu(lane).io.carry                                  := (selected && (!carryMask || !vm)) || merge && vm
    alu(lane).io.vxrm                                   := io.vxrm
    alu(lane).io.selector                               := rs1
    alu(lane).io.sew                                    := sew
    alu(lane).io.funct6                                 := op
    alu(lane).io.multiplyClass                          := multiply
    falu(lane).io.a                                     := a
    falu(lane).io.b                                     := Mux(fpReduction, accumulator, b)
    falu(lane).io.c                                     := operandReads(lane)._3
    falu(lane).io.sew                                   := sew
    falu(lane).io.op                                    := Mux(
      fpReduction,
      Mux(op === 5.U, 4.U, Mux(op === 7.U, 6.U, 0.U)),
      op
    )
    falu(lane).io.subop                                 := rs1
    falu(lane).io.roundingMode                          := io.roundingMode
    val fpRequired =
      laneSelected(lane) && !merge && (!fpReduction || (lane == 0).B)
    falu(
      lane
    ).io.start       := state === execute && instructionLegal && fp && fpRequired && !fpStarted(
      lane
    ) && falu(
      lane
    ).io.ready
    fpComplete(lane) := !fpRequired || fpDone(lane) || falu(lane).io.valid
    laneFlags(lane)  := Mux(
      merge,
      0.U,
      Mux(fpDone(lane), fpFlags(lane), falu(lane).io.flags)
    )
    laneLegal(lane)  := merge || Mux(
      fpDone(lane),
      fpLegal(lane),
      falu(lane).io.legal
    )
    val floatingValue = Mux(fpDone(lane), fpData(lane), falu(lane).io.result)
    val gathered      = Mux(sourceIndex < vlmax, a, 0.U)
    val slid          = Mux(
      slideOne && Mux(up, index === 0.U, index === vl - 1.U),
      scalar,
      Mux(!up && sourceIndex >= vlmax, 0.U, a)
    )
    val maskA         = maskElement(sourceWords(lane), index)
    val maskB         = maskElement(operandWords(lane), index)
    val maskResult    = MuxLookup(op, false.B)(
      Seq(
        24.U -> (maskA && !maskB),
        25.U -> (maskA && maskB),
        26.U -> (maskA || maskB),
        27.U -> (maskA ^ maskB),
        28.U -> (maskA || !maskB),
        29.U -> !(maskA && maskB),
        30.U -> !(maskA || maskB),
        31.U -> !(maskA ^ maskB)
      )
    )
    laneValue(lane) := Mux(
      fp,
      Mux(merge, Mux(vm || selected, b, a), floatingValue),
      Mux(
        wholeMove,
        a,
        Mux(
          maskLogical,
          maskResult,
          Mux(
            identity,
            index,
            Mux(
              slide || slideOne,
              slid,
              Mux(gather, gathered, alu(lane).io.result)
            )
          )
        )
      )
    )
    when(falu(lane).io.start)(fpStarted(lane) := true.B)
    when(state === execute && falu(lane).io.valid) {
      fpDone(lane)  := true.B
      fpData(lane)  := falu(lane).io.result
      fpFlags(lane) := falu(lane).io.flags
      fpLegal(lane) := falu(lane).io.legal
    }
  }

  val destinationGroupLegal =
    compare || maskLogical || population || first || reduction || scalarMoveOut || scalarMoveIn ||
      groupLegal(rd, destinationSew)

  val sourceGroupLegal =
    maskLogical || population || first || scalarMoveIn || scalarMoveOut || groupLegal(
      rs2,
      sourceSew
    )

  val operandGroupLegal =
    !vectorOperand || op === 18.U || reduction || scalarMoveOut || maskLogical || population || first ||
      groupLegal(rs1, sew)

  val aluLegal = alu
    .map(_.io.legal)
    .reduce(_ && _) || maskLogical || identity || gather || slide || slideOne ||
    scalarMoveOut || scalarMoveIn || population || first || intReduction || wholeMove

  val fpOperationLegal = Seq(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 18, 19, 23, 24, 25, 27, 28, 29, 31, 32, 33, 36, 39,
    40, 41, 42, 43, 44, 45, 46, 47)
    .map(n => op === n.U)
    .reduce(_ || _)

  val ordinaryLegal =
    !vtype(
      63
    ) && sourceSew <= p.maxSew.U && destinationSew <= p.maxSew.U && destinationGroupLegal && sourceGroupLegal && operandGroupLegal &&
      (!(intWide || floatWide || intNarrow || floatNarrow || extension) ||
        overlapLegal(rd, destinationSew, rs2, sourceSew) &&
        (!vectorOperand || op === 18.U || overlapLegal(
          rd,
          destinationSew,
          rs1,
          sew
        ))) &&
      (vm || rd =/= 0.U || compare || maskLogical || population || first || reduction) &&
      (!merge || !vm || rs2 === 0.U) && (!maskLogical || vm) &&
      (!carryOperation || carryMask || !vm) &&
      (!(reduction || population || first) || vstart === 0.U) &&
      (!fp || sew >= 2.U && fpOperationLegal && io.roundingMode <= 4.U &&
        (scalarMoveOut || scalarMoveIn || merge || falu(0).io.legal) &&
        (op =/= 16.U || scalarMoveOut || scalarMoveIn) &&
        (op =/= 18.U && op =/= 19.U || kind === 1.U) && (!merge || kind === 5.U)) &&
      (!integer || aluLegal)

  val wholeMoveLegal = vm && vstart === 0.U && Seq(0, 1, 3, 7)
    .map(n => rs1 === n.U)
    .reduce(_ || _) &&
    (rd & rs1) === 0.U && (rs2 & rs1) === 0.U && rd +& rs1 < 32.U && rs2 +& rs1 < 32.U

  val memoryLegal = memWidth <= p.maxSew.U && memSew <= p.maxSew.U && Seq(
    0,
    5,
    6,
    7
  ).map(n => kind === n.U).reduce(_ || _) && !instruction(28) &&
    Mux(
      wholeMemory,
      vm && Seq(0, 1, 3, 7)
        .map(n => instruction(31, 29) === n.U)
        .reduce(_ || _) &&
        (rd & instruction(31, 29)) === 0.U && rd +& instruction(
          31,
          29
        ) < 32.U && (!store || memWidth === 0.U),
      !vtype(63) && instruction(
        31,
        29
      ) === 0.U && (memMode =/= 0.U || rs2 === 0.U) &&
        groupLegal(rd, memSew) && (!indexed || groupLegal(
          rs2,
          memWidth
        )) && (vm || store || rd =/= 0.U)
    )

  val legal = Mux(
    memory,
    memoryLegal,
    Mux(wholeMove, wholeMoveLegal, (integer || fp) && ordinaryLegal)
  )

  instructionLegal := legal

  val required  = Wire(Vec(p.memoryPorts, Bool()))
  val addresses = Wire(Vec(p.memoryPorts, UInt(64.W)))
  val canceled  = Wire(Vec(p.memoryPorts, Bool()))
  for (port <- 0 until p.memoryPorts) {
    val index = base + port.U
    required(
      port
    )               := (port < p.laneNumber).B && index < limit && (wholeMemory || vm || maskElement(
      maskWords(port),
      index
    ))
    addresses(port) := scalar1 + Mux(
      indexed && !wholeMemory,
      operandReads(port)._1,
      Mux(memMode === 2.U, index * scalar2, index << memSew)
    )
    val earlierFailure = (0 until port)
      .map { prior =>
        required(prior) && (memoryFailed(prior) || io
          .memoryResponse(prior)
          .fire && io.memoryResponse(prior).bits.error)
      }
      .foldLeft(false.B)(_ || _)
    val earlierPending = (0 until port)
      .map { prior =>
        val distance        = addresses(port) - addresses(prior)
        val reverseDistance = addresses(prior) - addresses(port)
        val collision       =
          store && (distance < (1.U << memSew) || reverseDistance < (1.U << memSew))
        required(prior) && (memMode === 3.U || collision) && !received(prior)
      }
      .foldLeft(false.B)(_ || _)
    canceled(port) := !issued(port) && !offered(port) && earlierFailure
    io.memoryRequest(port)
      .valid                                         := state === execute && memory && legal && required(
      port
    ) && !issued(
      port
    ) && !earlierPending && !canceled(port)
    io.memoryRequest(port).bits.address              := addresses(port)
    io.memoryRequest(port).bits.write                := store
    io.memoryRequest(port).bits.data                 := operandReads(port)._3
    io.memoryRequest(port).bits.mask                 := MuxLookup(memSew, 255.U(8.W))(
      Seq(0.U -> 1.U, 1.U -> 3.U, 2.U -> 15.U)
    )
    io.memoryRequest(port).bits.size                 := memSew
    io.memoryResponse(port).ready                    := state === execute && memory && issued(
      port
    ) && !received(port)
    when(io.memoryRequest(port).valid)(offered(port) := true.B)
    when(io.memoryRequest(port).fire)(issued(port)   := true.B)
    when(io.memoryResponse(port).fire) {
      received(port)     := true.B
      memoryData(port)   := io.memoryResponse(port).bits.data
      memoryFailed(port) := io.memoryResponse(port).bits.error
    }
  }

  val memoryComplete = (0 until p.memoryPorts)
    .map(i => !required(i) || canceled(i) || received(i) || io.memoryResponse(i).fire)
    .reduce(_ && _)

  val failures = VecInit(
    (0 until p.memoryPorts).map(i =>
      required(i) && (memoryFailed(i) || io
        .memoryResponse(i)
        .fire && io.memoryResponse(i).bits.error)
    )
  )

  val failureIndex = PriorityEncoder(failures.asUInt)

  when(state === execute) {
    when(configure) {
      val immediate     = instruction(31, 30) === 3.U
      val registerType  = instruction(31, 25) === 64.U
      val requested     = Mux(
        registerType,
        scalar2,
        Mux(immediate, instruction(29, 20), instruction(30, 20))
      )
      val requestedSew  = requested(5, 3)
      val requestedLmul = requested(2, 0)
      val typeLegal     =
        requested(63, 8) === 0.U && requestedSew <= p.maxSew.U && requestedLmul =/= 4.U &&
          (!requestedLmul(
            2
          ) || (8.U << requestedSew) <= (p.eLen.U >> (8.U - requestedLmul)))
      val newMaximum    = maximum(requestedSew, requestedLmul)
      val keepLength    = !immediate && rs1 === 0.U && rd === 0.U
      when(
        !(immediate || registerType || !instruction(
          31
        )) || typeLegal && keepLength && newMaximum =/= vlmax
      ) {
        finish(true.B, 2.U, instruction)
      }.otherwise {
        val avl    = Mux(
          immediate,
          rs1,
          Mux(rs1 =/= 0.U, scalar1, Mux(rd =/= 0.U, newMaximum, vl))
        )
        val length = Mux(typeLegal, Mux(avl < newMaximum, avl, newMaximum), 0.U)
        vl                 := length
        vtype              := Mux(typeLegal, requested, "h8000000000000000".U)
        result.scalarWrite := rd =/= 0.U
        result.scalarData  := length
        finish()
      }
    }.elsewhen(!legal) {
      finish(true.B, 2.U, instruction)
    }.elsewhen(scalarMoveOut) {
      val value = element(sourceWords(0), 0.U, sew)
      when(fp) {
        result.floatWrite := true.B
        result.floatData  := Mux(
          sew === 2.U,
          Cat("hffffffff".U(32.W), value(31, 0)),
          value
        )
      }.otherwise {
        result.scalarWrite := rd =/= 0.U
        result.scalarData  := MuxLookup(sew, value)(
          Seq(
            0.U -> Cat(Fill(56, value(7)), value(7, 0)),
            1.U -> Cat(Fill(48, value(15)), value(15, 0)),
            2.U -> Cat(Fill(32, value(31)), value(31, 0))
          )
        )
      }
      finish()
    }.elsewhen(scalarMoveIn) {
      when(vl =/= 0.U && vstart < vl) {
        val scalar = Mux(
          sew === 2.U && !floatScalar(63, 32).andR,
          "h7fc00000".U,
          floatScalar
        )
        writeElement(
          0,
          0.U,
          sew,
          Mux(fp, scalar, scalar1)
        )
      }
      finish()
    }.elsewhen(base >= limit) {
      when(population || first) {
        result.scalarWrite := rd =/= 0.U
        result.scalarData  := Mux(first, "hffffffffffffffff".U, count)
      }
      finish()
    }.elsewhen(memory) {
      when(memoryComplete) {
        for (lane <- 0 until p.laneNumber) {
          when(
            load && required(
              lane
            ) && (!failures.asUInt.orR || lane.U < failureIndex)
          ) {
            writeElement(
              lane,
              base + lane.U,
              memSew,
              Mux(
                io.memoryResponse(lane).fire,
                io.memoryResponse(lane).bits.data,
                memoryData(lane)
              )
            )
          }
        }
        when(failures.asUInt.orR) {
          vstart := base + failureIndex
          finish(true.B, Mux(load, 5.U, 7.U), addresses(failureIndex))
        }.otherwise(nextBatch(p.laneNumber.U))
      }
    }.elsewhen(population || first) {
      val selected = VecInit((0 until p.laneNumber).map { lane =>
        val index = base + lane.U
        index < vl && maskElement(sourceWords(lane), index) &&
        (vm || maskElement(maskWords(lane), index))
      })
      val total    = count + PopCount(selected)
      count := total
      when(first && selected.asUInt.orR || base + p.laneNumber.U >= vl) {
        result.scalarWrite := rd =/= 0.U
        result.scalarData  := Mux(
          first,
          Mux(
            selected.asUInt.orR,
            base + PriorityEncoder(selected.asUInt),
            "hffffffff".U
          ),
          total
        )
        finish()
      }.otherwise {
        base      := base + p.laneNumber.U
        readPhase := 0.U
        state     := capture
      }
    }.elsewhen(intReduction) {
      // All integer reductions are associative at the destination width. Keep
      // the original leaf order and ignore masked lanes instead of serializing them.
      def combine(left: (Bool, UInt), right: (Bool, UInt)): (Bool, UInt) = {
        val (leftValid, a)  = left
        val (rightValid, b) = right
        val width           = 8.U(7.W) << sew
        val sa              = (a << (p.eLen.U - width))(p.eLen - 1, 0).asSInt
        val sb              = (b << (p.eLen.U - width))(p.eLen - 1, 0).asSInt
        val value           = MuxLookup(op, a + b)(Seq(
          1.U -> (a & b),
          2.U -> (a | b),
          3.U -> (a ^ b),
          4.U -> Mux(b < a, b, a),
          5.U -> Mux(sb < sa, b, a),
          6.U -> Mux(b > a, b, a),
          7.U -> Mux(sb > sa, b, a)
        ))
        (leftValid || rightValid, Mux(!leftValid, b, Mux(!rightValid, a, value)))
      }
      def tree(values: Seq[(Bool, UInt)]): (Bool, UInt) = {
        if (values.size == 1) values.head
        else {
          val (left, right) = values.splitAt(values.size / 2)
          combine(tree(left), tree(right))
        }
      }
      val reduced = tree(Seq(true.B -> accumulator) ++
        (0 until p.laneNumber).map(lane => laneSelected(lane) -> laneA(lane)))._2
      accumulator := reduced
      when(base + p.laneNumber.U >= vl)(writeElement(0, 0.U, sew, reduced))
      nextBatch(p.laneNumber.U)
    }.elsewhen(fpReduction) {
      when(fpComplete(0)) {
        when(laneSelected(0) && !laneLegal(0))(finish(true.B, 2.U, instruction))
          .otherwise {
            val value = Mux(laneSelected(0), laneValue(0), accumulator)
            accumulator                        := value
            when(laneSelected(0))(result.flags := result.flags | laneFlags(0))
            when(base + 1.U >= vl)(writeElement(0, 0.U, sew, value))
            nextBatch(1.U)
          }
      }
    }.elsewhen(divide) {
      when(!io.initialize) {
        when(!divBatchValid) {
          divActive     := laneSelected
          divBatchValid := true.B
        }.elsewhen((0 until p.laneNumber).map(i => !divActive(i) || divDone(i)).reduce(_ && _)) {
          for (lane <- 0 until p.laneNumber) {
            when(divActive(lane)) {
              writeElement(lane, base + lane.U, destinationSew, divData(lane))
            }
          }
          nextBatch(p.laneNumber.U)
        }
      }
    }.elsewhen(!fp || fpComplete.reduce(_ && _)) {
      val floatingLegal = (0 until p.laneNumber)
        .map(i => !laneSelected(i) || laneLegal(i))
        .reduce(_ && _)
      when(fp && !floatingLegal)(finish(true.B, 2.U, instruction)).otherwise {
        for (lane <- 0 until p.laneNumber) {
          when(laneSelected(lane)) {
            writeElement(
              lane,
              base + lane.U,
              destinationSew,
              laneValue(lane),
              compare || maskLogical
            )
          }
        }
        when(fp) {
          result.flags := result.flags | (0 until p.laneNumber)
            .map(i => Mux(laneSelected(i), laneFlags(i), 0.U))
            .reduce(_ | _)
        }
        when(integer && !wholeMove) {
          result.saturated := result.saturated || (0 until p.laneNumber)
            .map(i => laneSelected(i) && alu(i).io.saturated)
            .reduce(_ || _)
        }
        nextBatch(p.laneNumber.U)
      }
    }
  }
  when(state === execute && writes.map(_.valid).reduce(_ || _) && !io.initialize) {
    pendingWrites := writes
    resumeState   := Mux(finishRequested, idle, capture)
    state         := commit
    writeSent     := false.B
  }
  when(io.initialize) {
    state                := idle
    readSent             := false.B
    snapshotSent         := false.B
    writeSent            := false.B
    firstBatch           := true.B
    readPhase            := 0.U
    resultValid          := false.B
    vl                   := 0.U
    vtype                := "h8000000000000000".U
    vstart               := 0.U
    divBatchValid        := false.B
    divStarted.foreach(_ := false.B)
    divDone.foreach(_    := false.B)
    fpStarted.foreach(_  := false.B)
    fpDone.foreach(_     := false.B)
    offered.foreach(_    := false.B)
    issued.foreach(_     := false.B)
    received.foreach(_   := false.B)
  }
}
