package framework.rvv

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.rvv.configs.RvvParam

@instantiable
class VectorCore(val p: RvvParam) extends Module {

  @public
  val io = IO(new Bundle {
    val issue          = Flipped(Decoupled(new VectorIssue))
    val result         = Decoupled(new VectorResult)
    val memoryRequest  = Vec(p.memoryPorts, Decoupled(new VectorMemoryRequest))
    val memoryResponse = Vec(p.memoryPorts, Flipped(Decoupled(new VectorMemoryResponse)))
    val busy           = Output(Bool())
  })

  val active            = RegInit(false.B)
  val resultValid       = RegInit(false.B)
  val resultFault       = RegInit(false.B)
  val resultCause       = Reg(UInt(32.W))
  val resultValue       = Reg(UInt(32.W))
  val resultScalarWrite = RegInit(false.B)
  val resultScalarData  = Reg(UInt(32.W))
  val instruction       = Reg(UInt(32.W))
  val scalar1           = Reg(UInt(32.W))
  val scalar2           = Reg(UInt(32.W))
  val vl                = RegInit(0.U(32.W))
  val vtype             = RegInit("h80000000".U(32.W))
  val groupBase         = RegInit(0.U(log2Ceil(p.elementsPerRegister + p.laneNumber).W))
  val floatStarted      = RegInit(false.B)
  val requestIssued     = RegInit(VecInit(Seq.fill(p.memoryPorts)(false.B)))
  val responseReceived  = RegInit(VecInit(Seq.fill(p.memoryPorts)(false.B)))
  val responseData      = Reg(Vec(p.memoryPorts, UInt(32.W)))

  val opcode = instruction(6, 0)
  val rd     = instruction(11, 7)
  val funct3 = instruction(14, 12)
  val rs1    = instruction(19, 15)
  val rs2    = instruction(24, 20)
  val funct6 = instruction(31, 26)
  val vm     = instruction(25)

  val configure           = opcode === "h57".U && funct3 === 7.U
  val vectorLoad          = opcode === "h07".U
  val vectorStore         = opcode === "h27".U
  val memoryEncodingLegal = funct3 === 6.U && instruction(31, 26) === 0.U && rs2 === 0.U
  val integerOperation    = opcode === "h57".U &&
    (funct3 === 0.U || funct3 === 2.U || funct3 === 3.U || funct3 === 4.U || funct3 === 6.U)
  val floatOperation      = opcode === "h57".U && (funct3 === 1.U || funct3 === 5.U)

  val registers: Instance[VRF] = Instantiate(new VRF(p))
  registers.io.readAddress := VecInit(Seq(rs2, rs1, rd, 0.U))
  registers.io.write.valid := false.B
  registers.io.write.bits  := DontCare

  val integer:  Seq[Instance[IALU]] = Seq.fill(p.laneNumber)(Instantiate(new IALU))
  val floating: Seq[Instance[FALU]] = Seq.fill(p.laneNumber)(Instantiate(new FALU))
  val laneResult = Wire(Vec(p.laneNumber, UInt(32.W)))
  val laneWrite  = Wire(Vec(p.laneNumber, Bool()))

  for (lane <- 0 until p.laneNumber) {
    val element        = groupBase + lane.U
    val shift          = element << 5
    val source2        = (registers.io.readData(0) >> shift)(31, 0)
    val source1Vector  = (registers.io.readData(1) >> shift)(31, 0)
    val oldDestination = (registers.io.readData(2) >> shift)(31, 0)
    val mask           = (registers.io.readData(3) >> element)(0)
    val source1        = Mux(
      funct3 === 3.U,
      Cat(Fill(27, rs1(4)), rs1),
      Mux(funct3 === 4.U || funct3 === 5.U || funct3 === 6.U, scalar1, source1Vector)
    )

    integer(lane).io.a             := source2
    integer(lane).io.b             := source1
    integer(lane).io.c             := oldDestination
    integer(lane).io.carry         := false.B
    integer(lane).io.vxrm          := 0.U
    integer(lane).io.selector      := rs1
    integer(lane).io.sew           := 2.U
    integer(lane).io.funct6        := funct6
    integer(lane).io.multiplyClass := funct3 === 2.U || funct3 === 6.U

    floating(lane).io.a            := source2
    floating(lane).io.b            := source1
    floating(lane).io.c            := oldDestination
    floating(lane).io.op           := funct6
    floating(lane).io.subop        := rs1
    floating(lane).io.start        := active && floatOperation && !floatStarted
    floating(lane).io.roundingMode := 0.U

    laneResult(lane) := Mux(integerOperation, integer(lane).io.result, floating(lane).io.result)
    laneWrite(lane)  := element < vl && (vm || mask)
  }

  val writeData = (0 until p.laneNumber).map { lane =>
    val element = groupBase + lane.U
    val shift   = element << 5
    Mux(laneWrite(lane), (laneResult(lane).asUInt.pad(p.vLen) << shift)(p.vLen - 1, 0), 0.U)
  }.reduce(_ | _)

  val writeMask = (0 until p.laneNumber).map { lane =>
    val element = groupBase + lane.U
    val shift   = element << 5
    Mux(laneWrite(lane), ("hffffffff".U(p.vLen.W) << shift)(p.vLen - 1, 0), 0.U)
  }.reduce(_ | _)

  val lastGroup     = groupBase + p.laneNumber.U >= vl
  val integerLegal  = integer.map(_.io.legal).reduce(_ && _)
  val floatFinished = floating.map(_.io.valid).reduce(_ && _)

  io.issue.ready             := !active && !resultValid
  io.result.valid            := resultValid
  io.result.bits.fault       := resultFault
  io.result.bits.cause       := resultCause
  io.result.bits.tval        := resultValue
  io.result.bits.scalarWrite := resultScalarWrite
  io.result.bits.scalarData  := resultScalarData
  io.busy                    := active || resultValid

  for (port <- 0 until p.memoryPorts) {
    val used     = port < p.laneNumber
    val element  = groupBase + port.U
    val required = used.B && element < vl
    val source   = if (used) (registers.io.readData(0) >> (element << 5))(31, 0) else 0.U

    io.memoryRequest(
      port
    ).valid                             := active && (vectorLoad || vectorStore) && memoryEncodingLegal && required && !requestIssued(port)
    io.memoryRequest(port).bits.address := scalar1 + (element << 2)
    io.memoryRequest(port).bits.write   := vectorStore
    io.memoryRequest(port).bits.data    := source
    io.memoryResponse(port).ready       := active && (vectorLoad || vectorStore) && requestIssued(port) && !responseReceived(
      port
    )
  }

  val memoryComplete = (0 until p.memoryPorts).map { port =>
    val required = port.U < p.laneNumber.U && groupBase + port.U < vl
    !required || responseReceived(port) || io.memoryResponse(port).fire
  }.reduce(_ && _)

  val memoryError = (0 until p.memoryPorts).map { port =>
    io.memoryResponse(port).fire && io.memoryResponse(port).bits.error
  }.reduce(_ || _)

  when(io.result.fire) {
    resultValid := false.B
  }

  when(io.issue.fire) {
    instruction                := io.issue.bits.instruction
    scalar1                    := io.issue.bits.scalar1
    scalar2                    := io.issue.bits.scalar2
    groupBase                  := 0.U
    floatStarted               := false.B
    requestIssued.foreach(_    := false.B)
    responseReceived.foreach(_ := false.B)
    active                     := true.B
    resultScalarWrite          := false.B
  }

  when(active && configure) {
    val vsetvli         = !instruction(31)
    val vsetivli        = instruction(31, 30) === 3.U
    val vsetvl          = instruction(31, 25) === 64.U
    val requestedType   = Mux(vsetvl, scalar2, instruction(29, 20))
    val requestedLength = Mux(vsetivli, rs1, scalar1)
    val legal           = (vsetvli || vsetivli || vsetvl) &&
      requestedType(5, 3) === 2.U && requestedType(2, 0) === 0.U
    when(legal) {
      vl                := Mux(requestedLength < p.elementsPerRegister.U, requestedLength, p.elementsPerRegister.U)
      vtype             := requestedType
      active            := false.B
      resultValid       := true.B
      resultFault       := false.B
      resultCause       := 0.U
      resultValue       := 0.U
      resultScalarWrite := rd =/= 0.U
      resultScalarData  := Mux(requestedLength < p.elementsPerRegister.U, requestedLength, p.elementsPerRegister.U)
    }.otherwise {
      active      := false.B
      resultValid := true.B
      resultFault := true.B
      resultCause := 2.U
      resultValue := instruction
    }
  }

  when(active && (vectorLoad || vectorStore) && !memoryEncodingLegal) {
    active      := false.B
    resultValid := true.B
    resultFault := true.B
    resultCause := 2.U
    resultValue := instruction
  }

  when(active && integerOperation) {
    when(integerLegal && !vtype(31)) {
      registers.io.write.valid        := true.B
      registers.io.write.bits.address := rd
      registers.io.write.bits.data    := writeData
      registers.io.write.bits.mask    := writeMask
      when(lastGroup) {
        active      := false.B
        resultValid := true.B
        resultFault := false.B
        resultCause := 0.U
        resultValue := 0.U
      }.otherwise {
        groupBase := groupBase + p.laneNumber.U
      }
    }.otherwise {
      active      := false.B
      resultValid := true.B
      resultFault := true.B
      resultCause := 2.U
      resultValue := instruction
    }
  }

  when(active && floatOperation) {
    floatStarted := true.B
    when(floatFinished) {
      registers.io.write.valid        := true.B
      registers.io.write.bits.address := rd
      registers.io.write.bits.data    := writeData
      registers.io.write.bits.mask    := writeMask
      floatStarted                    := false.B
      when(lastGroup) {
        active      := false.B
        resultValid := true.B
        resultFault := false.B
        resultCause := 0.U
        resultValue := 0.U
      }.otherwise {
        groupBase := groupBase + p.laneNumber.U
      }
    }
  }

  for (port <- 0 until p.memoryPorts) {
    when(io.memoryRequest(port).fire) {
      requestIssued(port) := true.B
    }
    when(io.memoryResponse(port).fire) {
      responseReceived(port) := true.B
      responseData(port)     := io.memoryResponse(port).bits.data
    }
  }

  when(active && (vectorLoad || vectorStore) && memoryComplete) {
    when(memoryError) {
      active      := false.B
      resultValid := true.B
      resultFault := true.B
      resultCause := Mux(vectorLoad, 5.U, 7.U)
      resultValue := scalar1 + (groupBase << 2)
    }.otherwise {
      when(vectorLoad) {
        val loadData = (0 until p.laneNumber).map { lane =>
          val element = groupBase + lane.U
          val data    = Mux(io.memoryResponse(lane).fire, io.memoryResponse(lane).bits.data, responseData(lane))
          Mux(element < vl, (data.pad(p.vLen) << (element << 5))(p.vLen - 1, 0), 0.U)
        }.reduce(_ | _)
        val loadMask = (0 until p.laneNumber).map { lane =>
          val element = groupBase + lane.U
          Mux(element < vl, ("hffffffff".U(p.vLen.W) << (element << 5))(p.vLen - 1, 0), 0.U)
        }.reduce(_ | _)
        registers.io.write.valid := true.B
        registers.io.write.bits.address := rd
        registers.io.write.bits.data    := loadData
        registers.io.write.bits.mask    := loadMask
      }
      requestIssued.foreach(_    := false.B)
      responseReceived.foreach(_ := false.B)
      when(lastGroup) {
        active      := false.B
        resultValid := true.B
        resultFault := false.B
        resultCause := 0.U
        resultValue := 0.U
      }.otherwise {
        groupBase := groupBase + p.laneNumber.U
      }
    }
  }

  when(active && !configure && !integerOperation && !floatOperation && !vectorLoad && !vectorStore) {
    active      := false.B
    resultValid := true.B
    resultFault := true.B
    resultCause := 2.U
    resultValue := instruction
  }
}
