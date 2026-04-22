package examples.toy.configs

import upickle.default._

case class ToyConfig(nTiles: Int)

object ToyConfig {
  implicit val rw: ReadWriter[ToyConfig] = macroRW

  def apply(): ToyConfig = {
    val jsonStr = scala.io.Source.fromFile("src/main/scala/examples/toy/configs/default.json").mkString
    read[ToyConfig](jsonStr)
  }

}
