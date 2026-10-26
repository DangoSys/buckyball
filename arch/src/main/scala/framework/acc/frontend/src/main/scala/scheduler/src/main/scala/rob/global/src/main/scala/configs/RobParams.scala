package framework.acc.frontend.scheduler.rob.global.configs

import chisel3.util.log2Ceil
import upickle.default.{macroRW, ReadWriter}

case class RobParams(entries: Int, dataBits: Int, retireWidth: Int) {
  require(entries > 1 && dataBits > 0 && retireWidth > 0 && retireWidth <= entries)
  val idBits = log2Ceil(entries)
}

object RobParams { implicit val rw: ReadWriter[RobParams] = macroRW }
