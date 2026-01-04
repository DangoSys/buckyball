package framework.memdomain.configs

import upickle.default._

/**
 * MemDomain参数
 */
case class MemDomainParam(
  bankNum:                Int,
  bankWidth:              Int,
  bankEntries:            Int,
  bankMaskLen:            Int,
  tlb_size:               Int,
  dma_n_xacts:            Int,
  dma_maxbytes:           Int,
  bankChannel:            Int,
  max_in_flight_mem_reqs: Int,
  dma_buswidth:           Int,
  memAddrLen:             Int,
  tmaReadChannel:         Int,
  tmaWriteChannel:        Int)

object MemDomainParam {
  implicit val rw: ReadWriter[MemDomainParam] = macroRW

  def apply(): MemDomainParam = {
    val jsonStr = scala.io.Source.fromFile("src/main/scala/framework/memdomain/configs/default.json").mkString
    read[MemDomainParam](jsonStr)
  }

}
