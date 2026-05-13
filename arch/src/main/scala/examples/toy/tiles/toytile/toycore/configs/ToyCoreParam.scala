package examples.toy.tiles.toytile.toycore.configs

import upickle.default._
import framework.core.bbtile.configs.RocketCoreParam

case class ToyCoreConfig(
  balldomain: String,
  rocketCore: RocketCoreParam)

object ToyCoreConfig {
  implicit val rw: ReadWriter[ToyCoreConfig] = macroRW

  def apply(): ToyCoreConfig = {
    val jsonStr =
      scala.io.Source.fromFile("src/main/scala/examples/toy/tiles/toytile/toycore/configs/default.json").mkString
    read[ToyCoreConfig](jsonStr)
  }

}
