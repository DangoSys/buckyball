package framework.balldomain.prototype.gemmini.configs

import upickle.default._

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
  val blockSize:    Int = totalRows // == totalColumns (must be square)
}

object GemminiBallParam {
  implicit val rw: ReadWriter[GemminiBallParam] = macroRW

  def apply(): GemminiBallParam = {
    val jsonStr = scala.io.Source.fromFile(
      "src/main/scala/framework/balldomain/prototype/gemmini/configs/default.json"
    ).mkString
    read[GemminiBallParam](jsonStr)
  }

}
