package framework.top.configs

import upickle.default._

case class TopConfig(
  ballMemChannelNum: Int,
  memBallChannelNum: Int)

object TopConfig {
  implicit val rw: ReadWriter[TopConfig] = macroRW

  def apply(): TopConfig = {
    val jsonStr = scala.io.Source.fromFile("src/main/scala/framework/top/configs/default.json").mkString
    read[TopConfig](jsonStr)
  }

}
