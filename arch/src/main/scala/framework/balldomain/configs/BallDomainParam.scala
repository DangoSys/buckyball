package framework.balldomain.configs

import upickle.default._

case class BallIdMapping(
  ballId:   Int,
  ballName: String,
  inBW:     Int,
  outBW:    Int)

case class BallDomainParam(
  ballNum:              Int,
  ballIdMappings:       Seq[BallIdMapping],
  bbusProducerChannels: Int,
  bbusConsumerChannels: Int)

object BallDomainParam {
  implicit val ballIdMappingRW: ReadWriter[BallIdMapping]   = macroRW
  implicit val rw:              ReadWriter[BallDomainParam] = macroRW

  def apply(): BallDomainParam = {
    val jsonStr = scala.io.Source.fromFile("src/main/scala/framework/balldomain/configs/default.json").mkString
    read[BallDomainParam](jsonStr)
  }

}
