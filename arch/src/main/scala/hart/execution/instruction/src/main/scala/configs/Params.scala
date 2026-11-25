package framework.hart.execution.instruction.configs

import upickle.default.{macroRW, ReadWriter}

case class Params(xLen: Int) { require(xLen == 32 || xLen == 64) }
object Params { implicit val rw: ReadWriter[Params] = macroRW }
