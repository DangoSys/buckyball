package framework.acc.frontend.scheduler.bat.configs

import upickle.default.{macroRW, ReadWriter}

case class Params(bankIdBits: Int, virtualBanks: Int, robEntries: Int) {
  require(bankIdBits > 0)
  require(virtualBanks > 0)
  require(robEntries > 1)
  require(2 * virtualBanks + robEntries <= (1 << bankIdBits))
}

object Params { implicit val rw: ReadWriter[Params] = macroRW }
