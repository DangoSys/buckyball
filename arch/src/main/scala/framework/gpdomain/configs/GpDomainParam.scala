package framework.gpdomain.configs

import upickle.default._

/**
 * GpDomain参数
 */
case class GpDomainParam(
  /** Number of lanes in the GP domain */
  laneNumber:   Int,
  /** Chaining size for instruction scheduling */
  chainingSize: Int,
  /** Vector length in bits */
  vLen:         Int,
  /** Data length per lane in bits */
  dLen:         Int,
  /** Element length in bits */
  eLen:         Int,
  /** Lane scale factor */
  laneScale:    Int)

object GpDomainParam {
  implicit val rw: ReadWriter[GpDomainParam] = macroRW

  /**
   * 从默认的局部JSON文件加载
   */
  def apply(): GpDomainParam = {
    val jsonStr = scala.io.Source.fromFile("src/main/scala/framework/gpdomain/configs/default.json").mkString
    read[GpDomainParam](jsonStr)
  }

}
