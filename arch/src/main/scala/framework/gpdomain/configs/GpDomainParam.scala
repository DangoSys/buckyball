package framework.gpdomain.configs

import upickle.default._

/**
 * GpDomain Parameter
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

  def apply(): GpDomainParam = GpDomainParam(
    laneNumber = 0,
    chainingSize = 0,
    vLen = 0,
    dLen = 0,
    eLen = 0,
    laneScale = 0
  )

}
