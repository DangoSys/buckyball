package examples.balls.gemmini.configs

import framework.balldomain.configs.BallParamLoader
import framework.top.GlobalConfig

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
  private val ballName = "GemminiBall"

  def apply(b: GlobalConfig): GemminiBallParam = {
    val tbl = BallParamLoader.ballTable(b, ballName)
    GemminiBallParam(
      meshRows = BallParamLoader.int(tbl, "meshRows"),
      meshColumns = BallParamLoader.int(tbl, "meshColumns"),
      tileRows = BallParamLoader.int(tbl, "tileRows"),
      tileColumns = BallParamLoader.int(tbl, "tileColumns"),
      inputWidth = BallParamLoader.int(tbl, "inputWidth"),
      accWidth = BallParamLoader.int(tbl, "accWidth"),
      spatialOutputWidth = BallParamLoader.int(tbl, "spatialOutputWidth"),
      tileLatency = BallParamLoader.int(tbl, "tileLatency"),
      outputDelay = BallParamLoader.int(tbl, "outputDelay")
    )
  }
}
