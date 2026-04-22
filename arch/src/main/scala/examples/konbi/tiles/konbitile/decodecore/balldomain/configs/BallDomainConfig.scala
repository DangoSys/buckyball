package examples.konbi.tiles.konbitile.decodecore.balldomain.configs

import upickle.default._
import framework.balldomain.configs.{BallDomainParam, BallIdMapping}

object BallDomainConfig {
  private val path =
    "src/main/scala/examples/konbi/tiles/konbitile/decodecore/balldomain/configs/default.json"

  def apply(): BallDomainParam = {
    val jsonStr = scala.io.Source.fromFile(path).mkString
    read[BallDomainParam](jsonStr)
  }

}
