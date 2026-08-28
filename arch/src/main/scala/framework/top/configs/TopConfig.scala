package framework.top.configs

import upickle.default._

case class TopConfig(
  memBallChannelNum: Int,
  nCores:            Int)

object TopConfig {
  implicit val rw: ReadWriter[TopConfig] = macroRW

  /** Stub; real values come from ChipLoader (tile + derived nCores). */
  def apply(): TopConfig = TopConfig(memBallChannelNum = 0, nCores = 0)
}
