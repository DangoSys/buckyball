package framework.acc.memdomain.configs

import buckyball.config.{MemDomainConfig, SharedMemConfig}

import upickle.default._

/**
 * MemDomain Parameter
 */
case class MemDomainParam(
  sharedPorts:             Int,
  bankNum:                 Int,
  bankLayoutEnable:        Boolean,
  bankWidth:               Int,
  bankEntries:             Int,
  bankMaskLen:             Int,
  virtualBankCount:        Int,
  sharedEnable:            Boolean,
  sharedEntries:           Int,
  sharedBankEntries:       Int,
  sharedBankNum:           Int,
  sharedInputChannels:     Int,
  sharedDefaultGroupCount: Int,
  bankRequestPorts:        Int,
  dma_buswidth:            Int,
  memAddrLen:              Int,
  mmioEnable:              Boolean,
  mmioBankNum:             Int,
  mmioBankEntries:         Int,
  mmioBankWidth:           Int,
  mmioReadWidth:           Int) {

  if (sharedEnable) {
    require(sharedBankEntries >= 2 && sharedBankEntries <= 65536)
    require(sharedEntries == sharedBankNum * sharedBankEntries, "Shared pool size must equal banks times rows per bank")
  }

  // MMIO derived values
  val mmioBankBytes:   Int = mmioBankEntries * (mmioBankWidth / 8)
  val mmioTotalBytes:  Int = mmioBankNum * mmioBankBytes
  val vbankIdWidth:    Int = chisel3.util.log2Up(virtualBankCount)
  val maxGroupCount:   Int = math.max(bankNum, if (sharedEnable) sharedBankNum else bankNum)
  val groupCountWidth: Int = chisel3.util.log2Up(maxGroupCount + 1)
  val groupIdWidth:    Int = chisel3.util.log2Up(maxGroupCount)
}

object MemDomainParam {

  def fromProto(
    p:            MemDomainConfig,
    shared:       SharedMemConfig,
    virtualBanks: Int,
    participants: Int
  ): MemDomainParam = {
    require(p.hasBank && p.hasDma && p.hasMmio, "Memory domain configuration is incomplete")
    if (shared.getEnable) require(participants > 0 && shared.getInputChannels % participants == 0)
    val bank = p.getBank
    require(bank.hasLayoutEnable, "Bank layout policy is required")
    if (shared.getEnable) {
      require(shared.getBankEntries > 0 && shared.getEntries % shared.getBankEntries == 0)
      require(shared.getBankWidth == bank.getWidth, "Private and shared bank interfaces must agree")
    }
    val mmio = p.getMmio
    MemDomainParam(
      if (shared.getEnable) shared.getInputChannels / participants else 0,
      bank.getNum,
      bank.getLayoutEnable,
      bank.getWidth,
      bank.getEntries,
      bank.getMaskLen,
      virtualBanks,
      shared.getEnable,
      shared.getEntries,
      shared.getBankEntries,
      if (shared.getEnable) shared.getEntries / shared.getBankEntries else 0,
      shared.getInputChannels,
      shared.getDefaultGroupCount,
      bank.getRequestPorts,
      p.getDma.getBusWidth,
      p.getAddrLen,
      mmio.getEnable,
      mmio.getBankNum,
      mmio.getBankEntries,
      mmio.getBankWidth,
      mmio.getReadWidth
    )
  }

  implicit val rw: ReadWriter[MemDomainParam] = macroRW

}
