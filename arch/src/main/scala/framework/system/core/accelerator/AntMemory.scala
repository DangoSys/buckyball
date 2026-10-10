package framework.system.core.accelerator

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import memcore.bus.chi
import memcore.memory.interlock.{Params => TrackingParams}

/** Independent fair, single-owner cache-maintenance and page-table service arbiters. */
@instantiable
class AntMemory(ports: Int, tracking: TrackingParams, bus: chi.Params) extends Module {
  require(ports > 0)

  @public val io = IO(new Bundle {
    val clients    = Vec(ports, Flipped(new AntMemoryPort(tracking, bus)))
    val controller = new AntMemoryPort(tracking, bus)
  })

  io.controller.cpuAllow        := io.clients.map(_.cpuAllow).reduce(_ && _)
  io.controller.cpuProbeAllow   := io.clients.map(_.cpuProbeAllow).reduce(_ && _)
  io.clients.foreach(_.cpuQuery := io.controller.cpuQuery)

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
    io.controller.maintenance,
    io.controller.maintained
  )
  arbitrate(
    io.clients.map(_.pteRequest),
    io.clients.map(_.pteResponse),
    io.controller.pteRequest,
    io.controller.pteResponse
  )
}
