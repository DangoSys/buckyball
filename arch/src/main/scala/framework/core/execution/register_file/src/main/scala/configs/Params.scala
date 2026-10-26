package framework.core.execution.register_file.configs

import upickle.default.{macroRW, ReadWriter}

case class Params(xLen: Int) { require(xLen == 64) }
object Params { implicit val rw: ReadWriter[Params] = macroRW }
