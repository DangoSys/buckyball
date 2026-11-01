package framework.core.clink.configs

import framework.ip.bus.{axi4, chi}
import upickle.default.{macroRW, ReadWriter}

case class Params(
  harts:        Int,
  backing:      Boolean,
  requesterIds: Seq[Int],
  line:         chi.Params,
  uncached:     axi4.Params,
  ddr:          axi4.Params,
  service:      Boolean) {
  def requesters: Int = requesterIds.size
  require(
    requesterIds.distinct.size == requesterIds.size && requesterIds.forall(_ > 0),
    "Invalid CHI requester node IDs"
  )
  require(!(backing && requesters > 0), "A core exports either backing requests or a coherent requester")
}

object Params { implicit val rw: ReadWriter[Params] = macroRW }
