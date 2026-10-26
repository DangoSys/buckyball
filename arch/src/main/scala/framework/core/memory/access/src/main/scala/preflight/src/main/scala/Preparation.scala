package framework.core.memory.access.preflight

import framework.core.memory.access.preflight.configs.PmpParams
import framework.core.memory.access.mmu.configs.{Params => WalkerParam}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.core.memory.access.preflight.configs.PmpParams
import framework.core.memory.access.preflight.PmpEntry
import framework.core.memory.access.mmu.{PteRequest, PteResponse}
import framework.core.memory.access.PhysicalRegion
import framework.core.memory.access.preflight.{
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

/** Client boundary for an externally selected DMA address-preparation implementation. */
class PreparationPort(config: PreparationParams, permissionParam: PmpParams, walkerParam: WalkerParam) extends Bundle {
  implicit val permission: PmpParams = permissionParam

  val command      = Decoupled(new Command(config))
  val contextTag   = Input(UInt(config.idBits.W))
  val contextValid = Output(Bool())
  val contextId    = Output(UInt(config.idBits.W))
  val contextPmp   = Output(Vec(permission.entries, new PmpEntry(permissionParam)))
  val ranges       = Flipped(Decoupled(new PreparedSegment(config)))
  val ready        = Flipped(Decoupled(new MapReady(config)))
  val release      = Decoupled(new MapTag(config))
  val queries      = Output(Vec(2, new MapQuery(config)))
  val results      = Input(Vec(2, new MapResult(config)))
  val pteRequest   = Flipped(Decoupled(new PteRequest(walkerParam)))
  val pteResponse  = Decoupled(new PteResponse)
}

/** Uses the admission snapshot and retains translations until DMA and cache completion. */
@instantiable
class Preparation(val specification: configs.Config, walkerParam: WalkerParam) extends Module {
  val config          = specification.parameters
  val permissionParam = specification.permission
  implicit val permission: PmpParams = permissionParam

  @public
  val io = IO(Flipped(new PreparationPort(config, permissionParam, walkerParam)))

  val preflight: Instance[Preflight]   =
    Instantiate(new Preflight(config))
  val maps:      Instance[PreparedMap] =
    Instantiate(new PreparedMap(config))

  val check: Instance[PermissionCheck] =
    Instantiate(new PermissionCheck(specification))

  preflight.io.command.valid := io.command.valid && maps.io.reserve.ready && !reset.asBool
  preflight.io.command.bits  := io.command.bits
  maps.io.reserve.valid      := io.command.valid && preflight.io.command.ready && !reset.asBool
  maps.io.reserve.bits.id    := io.command.bits.id
  io.command.ready           := preflight.io.command.ready && maps.io.reserve.ready && !reset.asBool
  assert(preflight.io.command.fire === io.command.fire, "Preparation command reservation is not atomic")
  assert(maps.io.reserve.fire === io.command.fire, "Preparation map reservation is not atomic")

  val authorization = preflight.io.authorization
  io.contextTag         := authorization.bits.id
  check.io.request <> authorization
  check.io.contextValid := io.contextValid
  check.io.contextId    := io.contextId
  check.io.pmp          := io.contextPmp
  preflight.io.permission <> check.io.response
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
