package framework.acc.frontend.scheduler.scoreboard.configs

import upickle.default.{macroRW, ReadWriter}

case class Params(bankIdBits: Int, banks: Int, entries: Int) {
  require(bankIdBits > 0 && banks > 1 && banks <= (BigInt(1) << bankIdBits))
  require(entries > 1)
}

object Params { implicit val rw: ReadWriter[Params] = macroRW }
