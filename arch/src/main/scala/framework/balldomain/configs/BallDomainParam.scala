package framework.balldomain.configs

import upickle.default._

/**
 * Ball ID映射配置
 */
case class BallIdMapping(
  ballId:   Int,
  ballName: String)

/**
 * BallDomain参数
 */
case class BallDomainParam(
  emptyBallid:    Int,
  ballIdMappings: Seq[BallIdMapping])

object BallDomainParam {
  implicit val ballIdMappingRW: ReadWriter[BallIdMapping]   = macroRW
  implicit val rw:              ReadWriter[BallDomainParam] = macroRW

  def apply(): BallDomainParam = {
    val jsonStr = scala.io.Source.fromFile("arch/src/main/scala/framework/balldomain/configs/default.json").mkString
    read[BallDomainParam](jsonStr)
  }

}
