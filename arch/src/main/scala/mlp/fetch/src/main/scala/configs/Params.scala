package framework.mlp.fetch.configs

import chisel3.util._
import upickle.default.{macroRW, ReadWriter}

case class Params(instructionBytes: Int, lines: Int) {
  require(instructionBytes == 2 || instructionBytes == 4)
  require(lines >= 2 && isPow2(lines))
  val lanes: Int = 4 / instructionBytes
}

object Params { implicit val rw: ReadWriter[Params] = macroRW }
