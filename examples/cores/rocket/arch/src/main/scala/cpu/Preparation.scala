package hier.core.rocket

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.system.core.rocket.{CpuParams, HasCpuParameters}
import freechips.rocketchip.rocket.PMP
import memcore.bus.chi.rnf.{CacheAccess, CacheResult}
import memcore.memory.cpu.PhysicalRegion
import memcore.memory.preflight.{
  Command,
  PreparedSegment,
  Preflight,
  PreparedMap,
  MapQuery,
  MapResult,
  MapReady,
  MapTag,
  Params => PreparationParams
}

/** Uses the admission snapshot and retains translations until DMA and cache completion. */
@instantiable
class Preparation(config: PreparationParams, regions: Seq[PhysicalRegion])(implicit val cpuParams: CpuParams)
    extends Module
    with HasCpuParameters {

  @public
  val io = IO(new Bundle {
    val command      = Flipped(Decoupled(new Command(config)))
    // AdmissionBridge owns the frozen context. Query it by tag instead of duplicating its table.
    val contextTag   = Output(UInt(config.idBits.W))
    val contextValid = Input(Bool())
    val contextId    = Input(UInt(config.idBits.W))
    val contextPmp   = Input(Vec(nPMPs, new PMP))
    // Every successful segment is delivered both to the retained map and the physical interlock.
    // A terminal error goes to the caller too, which must cancel its unsealed reservation.
    val ranges       = Decoupled(new PreparedSegment(config))
    val ready        = Decoupled(new MapReady(config))
    val release      = Flipped(Decoupled(new MapTag(config)))
    val queries      = Input(Vec(2, new MapQuery(config)))
    val results      = Output(Vec(2, new MapResult(config)))
    val pteRequest   = Decoupled(new CacheAccess(config.bus))
    val pteResponse  = Flipped(Decoupled(new CacheResult))
  })

  val preflight:  Instance[Preflight]       = Instantiate(new Preflight(config))
  val maps:       Instance[PreparedMap]     = Instantiate(new PreparedMap(config))
  val permission: Instance[PermissionCheck] = Instantiate(new PermissionCheck(config, regions))
  preflight.io.command.valid := io.command.valid && maps.io.reserve.ready && !reset.asBool
  preflight.io.command.bits  := io.command.bits
  maps.io.reserve.valid      := io.command.valid && preflight.io.command.ready && !reset.asBool
  maps.io.reserve.bits.id    := io.command.bits.id
  io.command.ready           := preflight.io.command.ready && maps.io.reserve.ready && !reset.asBool
  assert(preflight.io.command.fire === io.command.fire, "Preparation command reservation is not atomic")
  assert(maps.io.reserve.fire === io.command.fire, "Preparation map reservation is not atomic")

  val authorization = preflight.io.authorization
  io.contextTag              := authorization.bits.id
  permission.io.request <> authorization
  permission.io.contextValid := io.contextValid
  permission.io.contextId    := io.contextId
  permission.io.pmp          := io.contextPmp
  preflight.io.permission <> permission.io.response
  io.pteRequest <> preflight.io.pteRequest
  preflight.io.pteResponse <> io.pteResponse

  maps.io.prepared.valid      := preflight.io.prepared.valid && io.ranges.ready
  maps.io.prepared.bits       := preflight.io.prepared.bits
  io.ranges.valid             := preflight.io.prepared.valid && maps.io.prepared.ready
  io.ranges.bits              := preflight.io.prepared.bits
  preflight.io.prepared.ready := maps.io.prepared.ready && io.ranges.ready
  io.ready <> maps.io.ready
  maps.io.queries             := io.queries
  io.results                  := maps.io.results
  maps.io.release <> io.release
}
