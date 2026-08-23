package framework.balldomain.configs

import upickle.default._

case class BallIdMapping(
  ballId:        Int,
  ballName:      String,
  ballClass:     String,
  config:        Option[String] = None,
  inBW:          Int,
  outBW:         Int,
  mmioReadBW:    Int = 0,
  mmioWriteBW:   Int = 0,
  configBaseDir: String = "")

case class BallISAEntry(
  mnemonic: String,
  funct7:   Int,
  bid:      Int)

case class BallDomainParam(
  ballNum:        Int,
  ballIdMappings: Seq[BallIdMapping],
  ballISA:        Seq[BallISAEntry]) {

  require(ballNum == ballIdMappings.length, "ballNum must match ballIdMappings length")
  require(
    ballIdMappings.map(_.ballId).distinct.length == ballIdMappings.length,
    "ballIdMappings contains duplicate ballId"
  )
  require(
    ballISA.map(_.mnemonic).distinct.length == ballISA.length,
    "ballISA contains duplicate mnemonic"
  )
  require(
    ballISA.map(_.funct7).distinct.length == ballISA.length,
    "ballISA contains duplicate funct7"
  )
  require(
    ballISA.forall(entry => entry.funct7 >= 0 && entry.funct7 < 128),
    "ballISA funct7 must fit the 7-bit instruction field"
  )
  require(
    ballISA.forall(entry => ballIdMappings.exists(_.ballId == entry.bid)),
    "ballISA bid must reference a configured ballId"
  )

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
