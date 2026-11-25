package framework.hart.hlink.configs

import framework.generic.bus.{axi4, chi}
import upickle.default.{macroRW, ReadWriter}

case class Params(
  harts:        Int,
  backing:      Boolean,
  requesterIds: Seq[Int],
  line:         chi.Params,
  uncached:     axi4.Params,
  ddr:          axi4.Params,
  service:      Boolean,
  // The hart's thread DMA asks the device Home whether a page has cached lines.
  pages:        Boolean) {
  def requesters: Int = requesterIds.size
  require(
    requesterIds.distinct.size == requesterIds.size && requesterIds.forall(_ > 0),
    "Invalid CHI requester node IDs"
  )
  require(!(backing && requesters > 0), "A hart exports either backing requests or a coherent requester")
  require(!(pages && backing), "Page queries go to an external Home")
}

object Params { implicit val rw: ReadWriter[Params] = macroRW }
