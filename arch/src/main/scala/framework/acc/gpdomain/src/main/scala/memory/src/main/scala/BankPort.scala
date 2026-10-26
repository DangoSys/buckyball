package framework.acc.gpdomain.memory

import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import framework.acc.gpdomain._
import framework.acc.gpdomain.execution.{VectorMemoryRequest, VectorMemoryResponse}
import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.acc.memdomain.{BankPort => BankAccessPort}
import framework.acc.gpdomain.loader.KernelContext

class BankRequest(frontendParam: FrontendParam, memParam: MemDomainParam) extends Bundle {
  val operation = new VectorMemoryRequest
  val context   = new KernelContext(frontendParam, memParam)
}

@instantiable
class BankPort(val frontendParam: FrontendParam, val memParam: MemDomainParam) extends Module {
  val ownerId    = frontendParam.ownerId
  val lineBytes  = memParam.bankWidth / 8
  val offsetBits = log2Ceil(lineBytes)
  require(memParam.bankWidth >= 64 && isPow2(
    memParam.bankWidth
  ))
  require(lineBytes <= 65536)
  require(memParam.bankMaskLen == lineBytes)

  @public
  val io = IO(new Bundle {
    val request  = Flipped(Decoupled(new BankRequest(frontendParam, memParam)))
    val response = Decoupled(new VectorMemoryResponse)
    val bank     = new BankAccessPort(frontendParam, memParam)
  })

  val idle :: readRequest :: readResponse :: writeRequest :: writeResponse :: complete :: Nil = Enum(6)
  val state                                                                                   = RegInit(idle)
  val request                                                                                 = Reg(new BankRequest(frontendParam, memParam))
  val consumed                                                                                = RegInit(0.U(4.W))
  val result                                                                                  = RegInit(0.U(64.W))
  val error                                                                                   = RegInit(false.B)

  val byteCount     = (1.U(4.W) << request.operation.size)(3, 0)
  val byteAddress   = request.operation.address(31, 0) +& consumed
  val lineOffset    = byteAddress(offsetBits - 1, 0)
  val remaining     = byteCount - consumed
  val lineRemaining = lineBytes.U - lineOffset
  val chunk         = Mux(remaining <= lineRemaining, remaining, lineRemaining)(3, 0)
  val nextConsumed  = consumed + chunk
  val lineAddress   = byteAddress >> offsetBits
  val readShifted   = io.bank.read.resp.bits.data >> (lineOffset << 3)
  val chunkMask     = ((1.U(65.W) << (chunk << 3)) - 1.U)(63, 0)
  val readChunk     = readShifted.pad(64)(63, 0) & chunkMask

  io.request.ready       := state === idle
  io.response.valid      := state === complete
  io.response.bits.data  := result
  io.response.bits.error := error
  val localBank   = request.operation.address(63, 32)
  val fromRead    = localBank < request.context.readGroups
  val parentBank  = Mux(fromRead, request.context.readBank, request.context.writeBank)
  val parentGroup = Mux(fromRead, localBank, localBank - request.context.readGroups)
  io.bank.bank_id             := parentBank
  io.bank.group_id            := parentGroup
  io.bank.rob_id              := request.context.rob_id
  io.bank.inst_id             := request.context.instId
  io.bank.thread_id           := ownerId.U
  io.bank.is_shared           := false.B // BankRouter resolves private/shared ownership from the alias.
  io.bank.read.req.valid      := state === readRequest
  io.bank.read.req.bits.addr  := lineAddress
  io.bank.read.resp.ready     := state === readResponse
  io.bank.write.req.valid     := state === writeRequest
  io.bank.write.req.bits.addr := lineAddress
  io.bank.write.req.bits.data := (request.operation.data >> (consumed << 3)).pad(
    memParam.bankWidth
  ) << (lineOffset << 3)
  io.bank.write.resp.ready    := state === writeResponse
  for (i <- 0 until lineBytes) {
    val sourceByte = consumed + i.U - lineOffset
    io.bank.write.req.bits.mask(i) := i.U >= lineOffset && i.U < lineOffset +& chunk && request.operation.mask(
      sourceByte(2, 0)
    )
  }

  when(io.request.fire) {
    request  := io.request.bits
    consumed := 0.U
    result   := 0.U
    val end     = io.request.bits.operation.address(31, 0) +& (1.U(33.W) << io.request.bits.operation.size)
    val invalid = io.request.bits.operation.address(
      63,
      32
    ) >= (io.request.bits.context.readGroups +& io.request.bits.context.writeGroups) ||
      (io.request.bits.operation.write && io.request.bits.operation.address(
        63,
        32
      ) < io.request.bits.context.readGroups) ||
      end > (BigInt(memParam.bankEntries) * lineBytes).U
    error := invalid
    state := Mux(invalid, complete, Mux(io.request.bits.operation.write, writeRequest, readRequest))
  }
  when(io.bank.read.req.fire) {
    state := readResponse
  }
  when(io.bank.read.resp.fire) {
    result   := result | (readChunk << (consumed << 3))
    consumed := nextConsumed
    state    := Mux(nextConsumed === byteCount, complete, readRequest)
  }
  when(io.bank.write.req.fire) {
    state := writeResponse
  }
  when(io.bank.write.resp.fire) {
    consumed := nextConsumed
    error    := !io.bank.write.resp.bits.ok
    state    := Mux(!io.bank.write.resp.bits.ok || nextConsumed === byteCount, complete, writeRequest)
  }
  when(io.response.fire) {
    state := idle
  }
}
