package framework.thread.acc.gpdomain.execution.vector

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.thread.acc.gpdomain.configs.GpDomainParam

@instantiable
class VRF(val gpParam: GpDomainParam) extends Module {
  val p                = gpParam.execution
  val wordBits         = p.wordBits
  val wordsPerRegister = p.vLen / wordBits
  val bankCount        = p.registerBanks
  val bankBits         = log2Ceil(bankCount)
  val addressBits      = log2Ceil(32 * wordsPerRegister)
  val bankDepth        = 32 * wordsPerRegister / bankCount
  require(isPow2(bankCount))

  @public
  val io = IO(new Bundle {
    val initialize = Input(Bool())
    val busy       = Output(Bool())

    val request = Vec(
      bankCount,
      Flipped(Decoupled(new Bundle {
        val address = UInt(addressBits.W)
        val write   = Bool()
        val data    = UInt(wordBits.W)
        val mask    = UInt(wordBits.W)
      }))
    )

    val response = Vec(bankCount, Decoupled(UInt(wordBits.W)))
  })

  val clearing     = RegInit(true.B)
  val clearAddress = RegInit(0.U(log2Ceil(bankDepth).W))
  val responded    = RegInit(VecInit(Seq.fill(bankCount)(false.B)))

  when(io.initialize) {
    clearing     := true.B
    clearAddress := 0.U
  }.elsewhen(clearing) {
    when(clearAddress === (bankDepth - 1).U) {
      clearing := false.B
    }.otherwise {
      clearAddress := clearAddress + 1.U
    }
  }

  io.busy := io.initialize || clearing || responded.asUInt.orR

  // Each bank is one single-port SRAM with bit write masks: a request a cycle, its response (the read word, or the
  // completion of a write) the next cycle; the requester is ready for it.
  for (bank <- 0 until bankCount) {
    val memory   = SyncReadMem(bankDepth, Vec(wordBits, Bool()))
    val request  = io.request(bank)
    val response = io.response(bank)

    request.ready := !io.initialize && !clearing
    if (bankBits > 0) when(request.fire)(assert(request.bits.address(bankBits - 1, 0) === bank.U))
    val address  = Mux(clearing, clearAddress, request.bits.address >> bankBits)
    val data     = Mux(clearing, 0.U(wordBits.W), request.bits.data)
    val mask     = Mux(clearing, Fill(wordBits, 1.U(1.W)), request.bits.mask)
    val write    = clearing || request.bits.write
    val enable   = !io.initialize && (clearing || request.fire)
    val readData = memory.readWrite(address, VecInit(data.asBools), mask.asBools, enable, write)

    responded(bank) := request.fire && !io.initialize
    response.valid  := responded(bank)
    response.bits   := readData.asUInt
    when(responded(bank))(assert(response.ready, "VRF responses are taken the cycle they come"))
  }
}
