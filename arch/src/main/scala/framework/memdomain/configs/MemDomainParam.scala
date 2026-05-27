package framework.memdomain.configs

import upickle.default._

/**
 * MemDomain Parameter
 *
 * Real values are injected by `TomlConfigLoader`. The no-arg `apply()` is a
 * stub used only by `GlobalConfig()` before TOML overrides land — matching
 * how `BallDomainParam()` returns an empty default.
 */
case class MemDomainParam(
  bankNum:                 Int,
  bankWidth:               Int,
  bankEntries:             Int,
  bankMaskLen:             Int,
  sharedEnable:            Boolean,
  sharedEntries:           Int,
  sharedInputChannels:     Int,
  sharedDefaultGroupCount: Int,
  tlb_size:                Int,
  dma_n_xacts:             Int,
  dma_maxbytes:            Int,
  bankChannel:             Int,
  max_in_flight_mem_reqs:  Int,
  dma_buswidth:            Int,
  memAddrLen:              Int,
  tmaReadChannel:          Int,
  tmaWriteChannel:         Int,
  mmioBankNum:             Int,
  mmioBankEntries:         Int,
  mmioBankWidth:           Int,
  mmioReadWidth:           Int) {

  // MMIO derived values
  val mmioBankBytes:  Int = mmioBankEntries * (mmioBankWidth / 8)
  val mmioTotalBytes: Int = mmioBankNum * mmioBankBytes
}

object MemDomainParam {
  implicit val rw: ReadWriter[MemDomainParam] = macroRW

  /** Stub default; real values come from TomlConfigLoader. */
  def apply(): MemDomainParam = MemDomainParam(
    bankNum = 0,
    bankWidth = 0,
    bankEntries = 0,
    bankMaskLen = 0,
    sharedEnable = false,
    sharedEntries = 0,
    sharedInputChannels = 0,
    sharedDefaultGroupCount = 0,
    tlb_size = 0,
    dma_n_xacts = 0,
    dma_maxbytes = 0,
    bankChannel = 0,
    max_in_flight_mem_reqs = 0,
    dma_buswidth = 0,
    memAddrLen = 0,
    tmaReadChannel = 0,
    tmaWriteChannel = 0,
    mmioBankNum = 0,
    mmioBankEntries = 0,
    mmioBankWidth = 0,
    mmioReadWidth = 0
  )

}
