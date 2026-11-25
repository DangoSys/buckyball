package framework.hart.execution.fpu.configs

import upickle.default.{macroRW, ReadWriter}

case class Params(xLen: Int, fLen: Int) { require((xLen == 32 || xLen == 64) && (fLen == 32 || fLen == 64)) }
object Params { implicit val rw: ReadWriter[Params] = macroRW }
