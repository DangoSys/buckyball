package framework.core.execution.fetch.configs

import chisel3.util._
import upickle.default.{macroRW, ReadWriter}

case class Params(instructionBytes: Int) {
  require(instructionBytes == 2 || instructionBytes == 4)
  val lanes: Int = 4 / instructionBytes
}

object Params { implicit val rw: ReadWriter[Params] = macroRW }
