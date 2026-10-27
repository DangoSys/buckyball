package examples.balls.gemmini.configs

import buckyball.balls.gemmini.{Params => ParametersProto}

import upickle.default.{macroRW, ReadWriter}

case class GemminiBallParam(
  meshRows:           Int,
  meshColumns:        Int,
  tileRows:           Int,
  tileColumns:        Int,
  inputWidth:         Int,
  accWidth:           Int,
  spatialOutputWidth: Int,
  tileLatency:        Int,
  outputDelay:        Int) {

  val totalRows:    Int = meshRows * tileRows
  val totalColumns: Int = meshColumns * tileColumns
  val blockSize:    Int = totalRows
}

object GemminiBallParam {

  def fromProto(p: ParametersProto): GemminiBallParam = {
    require(p.getAllFields.size == p.getDescriptorForType.getFields.size, "Gemmini parameters are required")
    GemminiBallParam(
      p.getMeshRows,
      p.getMeshColumns,
      p.getTileRows,
      p.getTileColumns,
      p.getInputWidth,
      p.getAccWidth,
      p.getSpatialOutputWidth,
      p.getTileLatency,
      p.getOutputDelay
    )
  }

  implicit val rw: ReadWriter[GemminiBallParam] = macroRW
}
