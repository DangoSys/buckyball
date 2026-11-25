package framework.device.shm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.generic.queue.Queue
import framework.generic.memory.bank.{Bank, BankSetParams}

/** Device-owned SRAM. No thread mapping or hart bank state enters this module. */
@instantiable
class Storage(p: Params, clients: Int) extends Module {
  require(clients > 0)
  val bytes    = p.axi.bytes
  val count    = (p.capacity / p.bankBytes).toInt
  val geometry = BankSetParams(p.axi.dataBits, 1, p.bankBytes / bytes, math.max(1, log2Ceil(clients)))

  @public val io = IO(new Bundle {
    val access = Vec(clients, Flipped(new StorageAccess(p.axi.dataBits)))
    val busy   = Output(Bool())
  })

  val banks     = Seq.fill(count)(Instantiate(new Bank(geometry)))
  val occupied  = RegInit(VecInit(Seq.fill(clients)(0.U(3.W))))
  val responses = Seq.fill(clients)(Instantiate(new Queue(new StorageResponse(p.axi.dataBits), 4)))
  val cursor    = RegInit(0.U(math.max(1, log2Ceil(clients)).W))

  val candidates = VecInit(io.access.zipWithIndex.map { case (a, i) =>
    a.request.valid && (occupied(i) < 4.U || a.response.fire)
  })

  val later     = VecInit((0 until clients).map(i => candidates(i) && i.U >= cursor))
  val selected  = PriorityEncoder(Mux(later.asUInt.orR, later.asUInt, candidates.asUInt))
  val offered   = candidates.asUInt.orR && !reset.asBool
  val request   = io.access(selected).request.bits
  val bankIndex = (request.address >> log2Ceil(p.bankBytes))(math.max(1, log2Ceil(count)) - 1, 0)
  val ready     = VecInit(banks.map(_.io.request.ready))(bankIndex)
  val completed = VecInit(banks.map(_.io.response.valid))
  val result    = Mux1H(completed, banks.map(_.io.response.bits))
  val owner     = result.tag
  for ((bank, i) <- banks.zipWithIndex) {
    bank.io.request.valid      := offered && bankIndex === i.U
    bank.io.request.bits.addr  := (request.address >> log2Ceil(bytes))(geometry.rowBits - 1, 0)
    bank.io.request.bits.write := request.write
    bank.io.request.bits.data  := request.data
    bank.io.request.bits.mask  := request.mask
    bank.io.request.bits.tag   := selected
    bank.io.response.ready     := true.B
  }
  assert(PopCount(completed) <= 1.U, "Device banks reordered the request stream")

  for ((a, i) <- io.access.zipWithIndex) {
    a.request.ready                := offered && ready && selected === i.U
    a.response <> responses(i).io.deq
    responses(i).io.enq.valid      := completed.asUInt.orR && owner === i.U
    responses(i).io.enq.bits.data  := result.data
    responses(i).io.enq.bits.error := false.B
    occupied(i)                    := occupied(i) + a.request.fire - a.response.fire
    when(completed.asUInt.orR && owner === i.U) {
      assert(responses(i).io.enq.ready, "Device storage response credit exhausted")
    }
  }
  when(offered && ready) {
    assert(
      request.address % bytes.U === 0.U && request.address +& bytes.U <= p.capacity.U,
      "Device storage request is unaligned or outside its SRAM"
    )
    cursor           := Mux(selected === (clients - 1).U, 0.U, selected + 1.U)
  }
  io.busy := occupied.map(_ =/= 0.U).reduce(_ || _) || completed.asUInt.orR
}
