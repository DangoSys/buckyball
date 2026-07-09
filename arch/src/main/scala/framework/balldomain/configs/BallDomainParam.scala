package framework.balldomain.configs

import upickle.default._

case class BallIdMapping(
  ballId:    Int,
  ballName:  String,
  ballClass: String,
  config:    String,
  inBW:      Int,
  outBW:     Int)

case class BallISAEntry(
  mnemonic: String,
  funct7:   Int,
  bid:      Int)

case class BallDomainParam(
  ballNum:        Int,
  ballIdMappings: Seq[BallIdMapping],
  ballISA:        Seq[BallISAEntry]) {

  def mapping(ballName: String): BallIdMapping =
    ballIdMappings.find(_.ballName == ballName) match {
      case Some(m) => m
      case None    => throw new RuntimeException(s"No ballIdMapping for ballName=$ballName")
    }

}

object BallDomainParam {
  implicit val ballIdMappingRW: ReadWriter[BallIdMapping]   = macroRW
  implicit val ballISAEntryRW:  ReadWriter[BallISAEntry]    = macroRW
  implicit val rw:              ReadWriter[BallDomainParam] = macroRW

  /**
   * Empty default. Each example's GlobalConfig assembler is responsible for
   * supplying its own `BallDomainParam` (typically via a layered JSON loader).
   */
  def apply(): BallDomainParam = BallDomainParam(0, Seq.empty, Seq.empty)
}
