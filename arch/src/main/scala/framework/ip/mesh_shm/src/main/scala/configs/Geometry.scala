package framework.ip.memory.mesh_shm.configs

import upickle.default.{macroRW, ReadWriter}

case class Geometry(dataBits: Int, entriesPerBank: Int, banks: Int) {
  require(dataBits == 128 && entriesPerBank > 0 && banks > 0)
  val bankBytes: Int    = entriesPerBank * (dataBits / 8)
  val bytes:     BigInt = BigInt(banks) * bankBytes
}

object Geometry { implicit val rw: ReadWriter[Geometry] = macroRW }
