package framework.rvv

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.top.GlobalConfig

@instantiable
class VectorCore(val b: GlobalConfig) extends Module {
  private val p = b.rvv

  @public val io = IO(new Bundle {
    val initialize     = Input(Bool())
    val roundingMode   = Input(UInt(3.W))
    val vxrm           = Input(UInt(2.W))
    val vstartWrite    = Flipped(Valid(UInt(32.W)))
    val vl             = Output(UInt(32.W))
    val vtype          = Output(UInt(32.W))
    val vstart         = Output(UInt(32.W))
    val issue          = Flipped(Decoupled(new VectorIssue))
    val result         = Decoupled(new VectorResult)
    val memoryRequest  = Vec(p.memoryPorts, Decoupled(new VectorMemoryRequest))
    val memoryResponse =
      Vec(p.memoryPorts, Flipped(Decoupled(new VectorMemoryResponse)))
    val busy           = Output(Bool())
  })

  val idle :: capture :: execute :: Nil = Enum(3)
  val state                             = RegInit(idle)
  val resultValid                       = RegInit(false.B)
  val result                            = RegInit(0.U.asTypeOf(new VectorResult))
  val instruction                       = Reg(UInt(32.W))
  val scalar1                           = Reg(UInt(32.W))
  val scalar2                           = Reg(UInt(32.W))
  val floatScalar                       = Reg(UInt(64.W))
  val vl                                = RegInit(0.U(32.W))
  val vtype                             = RegInit("h80000000".U(32.W))
  val vstart                            = RegInit(0.U(32.W))
  val base                              = Reg(UInt(32.W))
  val source                            = Reg(Vec(32, Vec(p.vLen / 64, UInt(64.W))))
  val accumulator                       = Reg(UInt(64.W))
  val count                             = Reg(UInt(32.W))
  val offered                           = RegInit(VecInit(Seq.fill(p.memoryPorts)(false.B)))
  val issued                            = RegInit(VecInit(Seq.fill(p.memoryPorts)(false.B)))
  val received                          = RegInit(VecInit(Seq.fill(p.memoryPorts)(false.B)))
  val memoryData                        = Reg(Vec(p.memoryPorts, UInt(64.W)))
  val memoryFailed                      = RegInit(VecInit(Seq.fill(p.memoryPorts)(false.B)))
  val divBatchValid                     = RegInit(false.B)
  val divActive                         = RegInit(VecInit(Seq.fill(p.laneNumber)(false.B)))
  val divStarted                        = RegInit(VecInit(Seq.fill(p.laneNumber)(false.B)))
  val divDone                           = RegInit(VecInit(Seq.fill(p.laneNumber)(false.B)))
  val divData                           = Reg(Vec(p.laneNumber, UInt(64.W)))
  val fpStarted                         = RegInit(VecInit(Seq.fill(p.laneNumber)(false.B)))
  val fpDone                            = RegInit(VecInit(Seq.fill(p.laneNumber)(false.B)))
  val fpData                            = Reg(Vec(p.laneNumber, UInt(64.W)))
  val fpFlags                           = Reg(Vec(p.laneNumber, UInt(5.W)))
  val fpLegal                           = Reg(Vec(p.laneNumber, Bool()))

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

  def element(register: UInt, index: UInt, width: UInt): UInt = {
    val shift   = (index.pad(log2Ceil(8 * p.vLen)) << (width.pad(3) + 3.U))(
      log2Ceil(8 * p.vLen) - 1,
      0
    )
    val address = (register + (shift >> log2Ceil(p.vLen)))(4, 0)
    val word    = shift(log2Ceil(p.vLen) - 1, 6)
    val mask    = MuxLookup(width, "hffffffffffffffff".U(64.W))(
      Seq(0.U -> 255.U, 1.U -> 65535.U, 2.U -> "hffffffff".U)
    )
    (source(address)(word) >> shift(5, 0)) & mask
  }

  def maskElement(register: UInt, index: UInt): Bool = {
    val word = index(log2Ceil(p.vLen) - 1, 6)
    (source(register)(word) >> index(5, 0))(0)
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

  val vlmax     = maximum(sew, vtype(2, 0))
  val registers = Instantiate(new VRF(b))
  registers.io.initialize := io.initialize
  for (lane <- 0 until p.laneNumber) {
    registers.io.write(lane).valid := false.B
    registers.io.write(lane).bits  := 0.U.asTypeOf(registers.io.write(lane).bits)
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
      1.U(64.W),
      MuxLookup(width, "hffffffffffffffff".U(64.W))(
        Seq(0.U -> 255.U, 1.U -> 65535.U, 2.U -> "hffffffff".U)
      )
    )
    registers.io.write(lane).valid := true.B
    registers.io.write(lane).bits.address := rd + (bitOffset >> log2Ceil(
      p.vLen
    ))
    registers.io.write(lane).bits.word    := within(log2Ceil(p.vLen) - 1, 6)
    registers.io.write(lane).bits.mask    := (mask << within(5, 0))(63, 0)
    registers.io.write(lane).bits.data    := (data << within(5, 0))(63, 0)
  }

  def finish(
    fault: Bool = false.B,
    cause: UInt = 0.U,
    value: UInt = 0.U
  ): Unit = {
    state               := idle
    resultValid         := true.B
    result.fault        := fault
    result.cause        := cause
    result.tval         := value
    when(!fault)(vstart := 0.U)
  }

  def nextBatch(step: UInt): Unit = {
    when(base + step >= limit)(finish()).otherwise(base := base + step)
    offered.foreach(_                                   := false.B)
    issued.foreach(_                                    := false.B)
    received.foreach(_                                  := false.B)
    memoryFailed.foreach(_                              := false.B)
    divBatchValid                                       := false.B
    divStarted.foreach(_                                := false.B)
    divDone.foreach(_                                   := false.B)
    fpStarted.foreach(_                                 := false.B)
    fpDone.foreach(_                                    := false.B)
  }

  io.vl                             := vl
  io.vtype                          := vtype
  io.vstart                         := vstart
  io.busy                           := state =/= idle || resultValid
  io.issue.ready                    := state === idle && !resultValid && !io.initialize
  io.result.valid                   := resultValid
  io.result.bits                    := result
  when(io.result.fire)(resultValid  := false.B)
  when(io.vstartWrite.valid)(vstart := io.vstartWrite.bits)
  when(io.issue.fire) {
    instruction            := io.issue.bits.instruction
    scalar1                := io.issue.bits.scalar1
    scalar2                := io.issue.bits.scalar2
    floatScalar            := io.issue.bits.floating
    state                  := capture
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
  when(state === capture) {
    source      := registers.io.readWords
    accumulator := registers.io.readWords(rs1)(0) & MuxLookup(
      sew,
      "hffffffffffffffff".U(64.W)
    )(Seq(0.U -> 255.U, 1.U -> 65535.U, 2.U -> "hffffffff".U))
    state       := execute
  }

  val instructionLegal = Wire(Bool())
  val alu              = Seq.fill(p.laneNumber)(Instantiate(new IALU))
  val falu             = Seq.fill(p.laneNumber)(Instantiate(new FALU))
  val laneA            = Wire(Vec(p.laneNumber, UInt(64.W)))
  val laneB            = Wire(Vec(p.laneNumber, UInt(64.W)))
  val laneSelected     = Wire(Vec(p.laneNumber, Bool()))
  val laneValue        = Wire(Vec(p.laneNumber, UInt(64.W)))
  val laneFlags        = Wire(Vec(p.laneNumber, UInt(5.W)))
  val laneLegal        = Wire(Vec(p.laneNumber, Bool()))
  val fpComplete       = Wire(Vec(p.laneNumber, Bool()))
  val gather           =
    integer && !multiply && (op === 12.U || op === 14.U && kind === 0.U)
  val slide            =
    integer && !multiply && (op === 14.U || op === 15.U) && kind =/= 0.U
  val slideOne         = integer && multiply && (op === 14.U || op === 15.U)
  val readPortCount    = math.max(p.laneNumber, p.memoryPorts)

  val operandReads = (0 until readPortCount).map { port =>
    val index          = base + port.U
    val scalar         = Mux(kind === 3.U, Cat(Fill(59, rs1(4)), rs1), Cat(Fill(32, scalar1(31)), scalar1))
    val floatingScalar = Mux(sew === 2.U && !floatScalar(63, 32).andR, "h7fc00000".U, floatScalar)
    val b              =
      Mux(vectorOperand, element(rs1, index, Mux(gather && op === 14.U, 1.U, sew)), Mux(fp, floatingScalar, scalar))
    val slideOffset    = Mux(slideOne, 1.U, Mux(kind === 3.U, rs1, scalar1))
    val sourceIndex    = Mux(gather && !slide, b, Mux(op === 14.U, index - slideOffset, index + slideOffset))
    val permuteRead    = gather || slide || slideOne
    // element already keeps only the low group bit offset; upper index bits cannot affect its read.
    val a              = element(rs2, Mux(permuteRead, sourceIndex(31, 0), index), Mux(permuteRead, sew, sourceSew))
    val c              = element(rd, index, Mux(fp, sew, destinationSew))
    (a, b, c, sourceIndex)
  }

  for (lane <- 0 until p.laneNumber) {
    val index          = base + lane.U
    val selected       = maskElement(0.U, index)
    val scalar         = Mux(
      kind === 3.U,
      Cat(Fill(59, rs1(4)), rs1),
      Cat(Fill(32, scalar1(31)), scalar1)
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
    val maskA         = maskElement(rs2, index)
    val maskB         = maskElement(rs1, index)
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
    !vectorOperand || op === 18.U || reduction || scalarMoveOut || maskLogical ||
      groupLegal(rs1, sew)

  val aluLegal = alu
    .map(_.io.legal)
    .reduce(_ && _) || maskLogical || identity || gather || slide || slideOne ||
    scalarMoveOut || scalarMoveIn || population || first || intReduction || wholeMove

  val fpOperationLegal = Seq(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 18, 19, 23, 24, 25, 27, 28, 29, 31, 32, 33, 36, 39,
    40, 41, 42, 43, 44, 45, 46, 47)
    .map(n => op === n.U)
    .reduce(_ || _)

  val ordinaryLegal = !vtype(31) && sourceSew <= (if (p.eLen == 64) 3.U
                                                  else
                                                    2.U) && destinationSew <= 3.U &&
    destinationSew <= (if (p.eLen == 64) 3.U
                       else
                         2.U) && destinationGroupLegal && sourceGroupLegal && operandGroupLegal &&
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

  val memoryLegal = memWidth <= (if (p.eLen == 64) 3.U
                                 else 2.U) && memSew <= (if (p.eLen == 64) 3.U
                                                         else 2.U) && Seq(
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
      !vtype(31) && instruction(
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
  val addresses = Wire(Vec(p.memoryPorts, UInt(32.W)))
  val canceled  = Wire(Vec(p.memoryPorts, Bool()))
  for (port <- 0 until p.memoryPorts) {
    val index = base + port.U
    required(
      port
    )               := (port < p.laneNumber).B && index < limit && (wholeMemory || vm || maskElement(
      0.U,
      index
    ))
    addresses(port) := scalar1 + Mux(
      indexed && !wholeMemory,
      operandReads(port)._1(31, 0),
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
        requested(31, 8) === 0.U && requestedSew <= (if (p.eLen == 64) 3.U
                                                     else
                                                       2.U) && requestedLmul =/= 4.U &&
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
        vtype              := Mux(typeLegal, requested, "h80000000".U)
        result.scalarWrite := rd =/= 0.U
        result.scalarData  := length
        finish()
      }
    }.elsewhen(!legal) {
      finish(true.B, 2.U, instruction)
    }.elsewhen(scalarMoveOut) {
      val value = element(rs2, 0.U, sew)
      when(fp) {
        result.floatWrite := true.B
        result.floatData  := Mux(
          sew === 2.U,
          Cat("hffffffff".U(32.W), value(31, 0)),
          value
        )
      }.otherwise {
        result.scalarWrite := rd =/= 0.U
        result.scalarData  := MuxLookup(sew, value(31, 0))(
          Seq(
            0.U -> Cat(Fill(24, value(7)), value(7, 0)),
            1.U -> Cat(Fill(16, value(15)), value(15, 0))
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
          Mux(fp, scalar, Cat(Fill(32, scalar1(31)), scalar1))
        )
      }
      finish()
    }.elsewhen(base >= limit) {
      when(population || first) {
        result.scalarWrite := rd =/= 0.U
        result.scalarData  := Mux(first, "hffffffff".U, count)
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
        index < vl && maskElement(rs2, index) &&
        (vm || maskElement(0.U, index))
      })
      val total    = count + PopCount(selected)
      count            := total
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
      }.otherwise(base := base + p.laneNumber.U)
    }.elsewhen(intReduction) {
      var reduced: UInt = accumulator
      for (lane <- 0 until p.laneNumber) {
        val a     = laneA(lane)
        val width = 8.U(7.W) << sew
        val sa    = (a << (64.U - width))(63, 0).asSInt
        val sb    = (reduced << (64.U - width))(63, 0).asSInt
        val value = MuxLookup(op, reduced + a)(
          Seq(
            1.U -> (reduced & a),
            2.U -> (reduced | a),
            3.U -> (reduced ^ a),
            4.U -> Mux(a < reduced, a, reduced),
            5.U -> Mux(sa < sb, a, reduced),
            6.U -> Mux(a > reduced, a, reduced),
            7.U -> Mux(sa > sb, a, reduced)
          )
        )
        reduced = Mux(laneSelected(lane), value, reduced)
      }
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
  when(io.initialize) {
    state                := idle
    resultValid          := false.B
    vl                   := 0.U
    vtype                := "h80000000".U
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
