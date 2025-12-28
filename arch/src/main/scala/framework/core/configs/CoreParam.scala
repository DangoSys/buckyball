package framework.core.configs

import upickle.default._

/**
 * Core参数
 */
case class CoreParam(
  coreDataBytes: Int,
  xLen:          Int,
  vaddrBits:     Int,
  paddrBits:     Int,
  pgIdxBits:     Int)

object CoreParam {
  implicit val rw: ReadWriter[CoreParam] = macroRW

  def apply(): CoreParam = {
    val jsonStr = scala.io.Source.fromFile("arch/src/main/scala/framework/core/configs/default.json").mkString
    read[CoreParam](jsonStr)
  }

}
