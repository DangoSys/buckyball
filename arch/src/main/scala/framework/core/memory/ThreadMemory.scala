package framework.core.memory

import framework.core.memory.access.mmu.{PteRequest, PteResponse}
import framework.ip.memory.interlock.{Acknowledgement, CpuQuery, Maintenance}
import chisel3._
import framework.ip.memory.interlock.{Params => TrackingParam}
import framework.core.memory.access.mmu.configs.{Params => WalkerParam}
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

/** Thread-side cache services; requests retain their original admission tag. */
class MemoryPort(tracking: TrackingParam, walker: WalkerParam) extends Bundle {

  val cpuQuery    = Input(new CpuQuery(tracking))
  val cpuAllow    = Output(Bool())
  val maintenance = Decoupled(new Maintenance(tracking))
  val maintained  = Flipped(Decoupled(new Acknowledgement(tracking)))
  val pteRequest  = Decoupled(new PteRequest(walker))
  val pteResponse = Flipped(Decoupled(new PteResponse))
}

/** Independent fair, single-owner cache-maintenance and page-table service arbiters. */
@instantiable
class ThreadMemory(ports: Int, tracking: TrackingParam, walker: WalkerParam) extends Module {
  require(ports > 0)

  @public
  val io = IO(new Bundle {
    val clients = Vec(ports, Flipped(new MemoryPort(tracking, walker)))
    val system  = new MemoryPort(tracking, walker)
  })

  io.system.cpuAllow            := io.clients.map(_.cpuAllow).reduce(_ && _)
  io.clients.foreach(_.cpuQuery := io.system.cpuQuery)

  def arbitrate[T <: Data, R <: Data](
    requests:  Seq[DecoupledIO[T]],
    responses: Seq[DecoupledIO[R]],
    request:   DecoupledIO[T],
    response:  DecoupledIO[R]
  ): Unit = {
    val ownerBits  = math.max(1, log2Ceil(ports))
    val next       = RegInit(0.U(ownerBits.W))
    val owner      = Reg(UInt(ownerBits.W))
    val occupied   = RegInit(false.B)
    val sent       = RegInit(false.B)
    val packet     = Reg(chiselTypeOf(request.bits))
    val candidates = VecInit(requests.map(_.valid)).asUInt
    val later      = VecInit(requests.zipWithIndex.map { case (r, i) => r.valid && i.U >= next }).asUInt
    val selected   = PriorityEncoder(Mux(later.orR, later, candidates))
    for (i <- 0 until ports) {
      requests(i).ready  := !occupied && selected === i.U && !reset.asBool
      responses(i).valid := occupied && sent && owner === i.U && response.valid
      responses(i).bits  := response.bits
      when(requests(i).fire) {
        packet   := requests(i).bits
        owner    := i.U
        occupied := true.B
        sent     := false.B
        next     := (if (i + 1 == ports) 0 else i + 1).U
      }
    }
    request.valid := occupied && !sent && !reset.asBool
    request.bits            := packet
    when(request.fire)(sent := true.B)
    response.ready          := occupied && sent && VecInit(responses.map(_.ready))(owner) && !reset.asBool
    when(response.valid)(assert(occupied && sent, "Ant cache service returned a response without a sent owner"))
    when(response.fire) {
      occupied := false.B
      sent     := false.B
    }
  }

  arbitrate(
    io.clients.map(_.maintenance),
    io.clients.map(_.maintained),
    io.system.maintenance,
    io.system.maintained
  )
  arbitrate(
    io.clients.map(_.pteRequest),
    io.clients.map(_.pteResponse),
    io.system.pteRequest,
    io.system.pteResponse
  )
}
