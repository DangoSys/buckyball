package framework.balldomain.configs

import upickle.default._

/**
 * Ball ID映射配置
 */
case class BallIdMapping(
  ballId:   Int,
  ballName: String,
  inBW:     Int,
  outBW:    Int)

/**
 * BallDomain参数
 */
case class BallDomainParam(
  ballNum:        Int,
  ballIdMappings: Seq[BallIdMapping],
  bbusChannel:    Int)

object BallDomainParam {
  implicit val ballIdMappingRW: ReadWriter[BallIdMapping]   = macroRW
  implicit val rw:              ReadWriter[BallDomainParam] = macroRW

  def apply(): BallDomainParam = {
    val jsonStr = scala.io.Source.fromFile("src/main/scala/framework/balldomain/configs/default.json").mkString
    read[BallDomainParam](jsonStr)
  }

}
