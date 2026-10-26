package framework.core.execution.muldiv.configs

import upickle.default.{macroRW, ReadWriter}

case class Params(xLen: Int) { require(xLen == 64) }
object Params { implicit val rw: ReadWriter[Params] = macroRW }
