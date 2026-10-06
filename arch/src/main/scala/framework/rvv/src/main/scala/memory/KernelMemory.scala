package framework.rvv

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

@instantiable
class KernelMemory(val b: GlobalConfig) extends Module {
  private val p = b.rvv

  @public
  val io = IO(new Bundle {
    val request    = Vec(p.memoryPorts, Flipped(Decoupled(new VectorMemoryRequest)))
    val response   = Vec(p.memoryPorts, Decoupled(new VectorMemoryResponse))
    val buffer     = Input(Bool())
    val loading    = Input(Bool())
    val constBytes = Input(UInt(32.W))
  })

  val constants0                                      = SyncReadMem(p.constBytes / 8, Vec(8, UInt(8.W)))
  val constants1                                      = SyncReadMem(p.constBytes / 8, Vec(8, UInt(8.W)))
  val stack                                           = SyncReadMem(p.stackBytes / 8, Vec(8, UInt(8.W)))
  val pending                                         = RegInit(VecInit(Seq.fill(p.memoryPorts)(false.B)))
  val responseValid                                   = RegInit(VecInit(Seq.fill(p.memoryPorts)(false.B)))
  val responseData                                    = Reg(Vec(p.memoryPorts, UInt(64.W)))
  val responseError                                   = Reg(Vec(p.memoryPorts, Bool()))
  val idle :: writeHigh :: readLow :: readHigh :: Nil = Enum(4)
  val state                                           = RegInit(idle)
  val owner                                           = Reg(UInt(log2Ceil(p.memoryPorts).max(1).W))
  val saved                                           = Reg(new VectorMemoryRequest)
  val selectedConstant                                = Reg(Bool())
  val selectedBuffer                                  = Reg(Bool())
  val selectedCrossing                                = Reg(Bool())
  val low                                             = Reg(UInt(64.W))

  val writes    =
    VecInit((0 until p.memoryPorts).map(port => io.request(port).valid && !pending(port) && io.request(port).bits.write))
  val hasWrite  = writes.asUInt.orR
  val writePort = PriorityEncoder(writes)
  val arbiter   = Module(new RRArbiter(new VectorMemoryRequest, p.memoryPorts))
  arbiter.io.out.ready := state === idle
  for (port <- 0 until p.memoryPorts) {
    val eligible = !pending(port) && (!hasWrite || writePort === port.U)
    arbiter.io.in(port).valid    := io.request(port).valid && eligible
    arbiter.io.in(port).bits     := io.request(port).bits
    io.request(port).ready       := arbiter.io.in(port).ready && eligible
    io.response(port).valid      := responseValid(port)
    io.response(port).bits.data  := responseData(port)
    io.response(port).bits.error := responseError(port)
    when(io.response(port).fire) {
      pending(port)       := false.B
      responseValid(port) := false.B
    }
  }

  val request          = arbiter.io.out.bits
  val bytes            = 1.U(5.W) << request.size
  val end              = request.address +& bytes
  val constant         = request.address >= "h80000000".U && end <= (BigInt("80000000", 16) + p.constBytes).U
  val temporary        = request.address >= "h80001000".U && end <= (BigInt("80001000", 16) + p.stackBytes).U
  val constantReadable = constant && end <= ("h80000000".U(33.W) + io.constBytes)
  val legal            = Mux(io.loading, constant && request.write, temporary || (constantReadable && !request.write))
  val crossing         = (request.address(2, 0) +& bytes) > 8.U
  val active           = Mux(state === idle, request, saved)
  val activeConstant   = Mux(state === idle, constant, selectedConstant)
  val activeBuffer     = Mux(state === idle, io.buffer, selectedBuffer)
  val second           = state === writeHigh || (state === readLow && selectedCrossing)
  val line             = active.address(11, 3) + second
  val read             = (arbiter.io.out.fire && legal && !request.write) || (state === readLow && selectedCrossing)
  val write            = (arbiter.io.out.fire && legal && request.write) || state === writeHigh
  val data             = (active.data.pad(128) << (active.address(2, 0) << 3))(127, 0)
  val mask             = (active.mask.pad(16) << active.address(2, 0))(15, 0)
  val word             = Mux(second, data(127, 64), data(63, 0))
  val wordMask         = Mux(second, mask(15, 8), mask(7, 0))
  val writeData        = VecInit((0 until 8).map(byte => word(8 * byte + 7, 8 * byte)))
  val writeMask        = (0 until 8).map(byte => wordMask(byte))

  val constant0Enable = (read || write) && activeConstant && !activeBuffer
  val constant1Enable = (read || write) && activeConstant && activeBuffer
  val stackEnable     = (read || write) && !activeConstant
  val constant0Read   = constants0.readWrite(line, writeData, writeMask, constant0Enable, write)
  val constant1Read   = constants1.readWrite(line, writeData, writeMask, constant1Enable, write)
  val stackRead       = stack.readWrite(line, writeData, writeMask, stackEnable, write)
  val readData        = Mux(selectedConstant, Mux(selectedBuffer, constant1Read.asUInt, constant0Read.asUInt), stackRead.asUInt)

  when(arbiter.io.out.fire) {
    owner                      := arbiter.io.chosen
    saved                      := request
    selectedConstant           := constant
    selectedBuffer             := io.buffer
    selectedCrossing           := crossing
    pending(arbiter.io.chosen) := true.B
    when(!legal || (request.write && !crossing)) {
      responseValid(arbiter.io.chosen) := true.B
      responseData(arbiter.io.chosen)  := 0.U
      responseError(arbiter.io.chosen) := !legal
    }.otherwise {
      state := Mux(request.write, writeHigh, readLow)
    }
  }
  when(state === writeHigh) {
    responseValid(owner) := true.B
    responseData(owner)  := 0.U
    responseError(owner) := false.B
    state                := idle
  }
  when(state === readLow) {
    when(selectedCrossing) {
      low   := readData
      state := readHigh
    }.otherwise {
      responseValid(owner) := true.B
      responseData(owner)  := readData >> (saved.address(2, 0) << 3)
      responseError(owner) := false.B
      state                := idle
    }
  }
  when(state === readHigh) {
    responseValid(owner) := true.B
    responseData(owner)  := (Cat(readData, low) >> (saved.address(2, 0) << 3))(63, 0)
    responseError(owner) := false.B
    state                := idle
  }
}
