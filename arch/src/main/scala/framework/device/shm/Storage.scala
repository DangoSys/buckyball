package framework.device.shm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

/** Device-owned SRAM. No thread mapping or core bank state enters this module. */
@instantiable
class Storage(p: Params, clients: Int) extends Module {
  require(clients > 0)
  val bytes = p.axi.bytes
  val rows  = (p.capacity / bytes).toInt
  require(isPow2(rows))

  @public val io = IO(new Bundle {
    val access = Vec(clients, Flipped(new StorageAccess(p.axi.dataBits)))
    val busy   = Output(Bool())
  })

  val memory     = SyncReadMem(rows, Vec(bytes, UInt(8.W)))
  val occupied   = RegInit(VecInit(Seq.fill(clients)(false.B)))
  val responses  = Reg(Vec(clients, new StorageResponse(p.axi.dataBits)))
  val ready      = RegInit(VecInit(Seq.fill(clients)(false.B)))
  val cursor     = RegInit(0.U(math.max(1, log2Ceil(clients)).W))
  val candidates = VecInit(io.access.zipWithIndex.map { case (a, i) => a.request.valid && !occupied(i) })
  val later      = VecInit((0 until clients).map(i => candidates(i) && i.U >= cursor))
  val selected   = PriorityEncoder(Mux(later.asUInt.orR, later.asUInt, candidates.asUInt))
  val offered    = candidates.asUInt.orR && !reset.asBool
  val request    = io.access(selected).request.bits
  val row        = (request.address >> log2Ceil(bytes))(log2Ceil(rows) - 1, 0)
  val read       =
    memory.readWrite(row, request.data.asTypeOf(Vec(bytes, UInt(8.W))), request.mask.asBools, offered, request.write)
  val completed  = RegNext(offered, false.B)
  val owner      = RegEnable(selected, offered)
  val written    = RegEnable(request.write, offered)

  for ((a, i) <- io.access.zipWithIndex) {
    a.request.ready  := offered && selected === i.U
    a.response.valid := ready(i)
    a.response.bits  := responses(i)
    when(a.response.fire) { ready(i) := false.B; occupied(i) := false.B }
  }
  when(offered) {
    assert(
      request.address   % bytes.U === 0.U && request.address +& bytes.U <= p.capacity.U,
      "Device storage request is unaligned or outside its SRAM"
    )
    occupied(selected) := true.B
    cursor             := Mux(selected === (clients - 1).U, 0.U, selected + 1.U)
  }
  when(completed) {
    responses(owner).data  := Mux(written, 0.U, read.asUInt)
    responses(owner).error := false.B
    ready(owner)           := true.B
  }
  io.busy := occupied.asUInt.orR || completed
}
