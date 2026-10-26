package framework.acc.balldomain.configs

import buckyball.config.{BallDomain => BallDomainProto, BallInstance => BallInstanceProto}
import scala.jdk.CollectionConverters._
import com.google.protobuf.{Any => BallParameters}

import upickle.default.{macroRW, readwriter, ReadWriter}
import BallRouteParam.parametersRW

case class InstructionParam(mnemonic: String, funct7: Int) {
  require(funct7 >= 0 && funct7 < 128, "funct7 must fit the 7-bit instruction field")
}

object InstructionParam { implicit val rw: ReadWriter[InstructionParam] = macroRW }

case class BallRouteParam(
  id:           Int,
  name:         String,
  inBW:         Int,
  outBW:        Int,
  mmioReadBW:   Int,
  mmioWriteBW:  Int,
  instructions: Seq[InstructionParam],
  component:    String,
  parameters:   BallParameters,
  rtlClass:     String) {
  require(inBW > 0 && outBW > 0, "BBus widths must be positive")
  require(id >= 0 && id < 32, "Ball ID must fit the 5-bit target field")
  require(
    instructions.map(_.mnemonic).distinct.size == instructions.size,
    "Duplicate instruction mnemonic in Ball instance"
  )
}

object BallRouteParam {

  def fromProto(source: BallInstanceProto): BallRouteParam = BallRouteParam(
    source.getBallId,
    source.getBallName,
    source.getInBw,
    source.getOutBw,
    source.getMmioReadBw,
    source.getMmioWriteBw,
    source.getInstructionsList.asScala.map(i => InstructionParam(i.getMnemonic, i.getFunct7)).toSeq,
    source.getBallDir,
    source.getParameters,
    source.getRtlClass
  )

  implicit val parametersRW: ReadWriter[BallParameters] =
    readwriter[Array[Byte]].bimap[BallParameters](_.toByteArray, bytes => BallParameters.parseFrom(bytes))
  implicit val rw:           ReadWriter[BallRouteParam] = macroRW
}

case class BallDomainParam(balls: Seq[BallRouteParam]) {
  require(balls.map(_.id).distinct.size == balls.size, "Duplicate Ball ID")
  require(balls.map(_.name).distinct.size == balls.size, "Duplicate Ball instance name")

  val instructions = balls.flatMap(_.instructions)
  require(instructions.map(_.funct7).distinct.size == instructions.size, "Duplicate instruction encoding")
}

object BallDomainParam {
  def fromProto(p: BallDomainProto): BallDomainParam =
    BallDomainParam(p.getBallsList.asScala.map(BallRouteParam.fromProto).toSeq)
  implicit val rw: ReadWriter[BallDomainParam] = macroRW
}
