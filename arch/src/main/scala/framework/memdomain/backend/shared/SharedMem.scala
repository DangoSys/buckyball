package framework.memdomain.backend.shared

import framework.top.GlobalConfig

object SharedMemLayout {

  def totalBank(b: GlobalConfig): Int = {
    require(b.memDomain.sharedEnable, "shared memory is disabled")
    require(b.memDomain.sharedBankNum > 0, "sharedBankNum must be > 0")
    b.memDomain.sharedBankNum
  }

  def channelPerHart(b: GlobalConfig): Int = {
    if (!b.memDomain.sharedEnable) {
      return 0
    }
    val computeCores = b.memDomain.computeCoreIds.size
    require(computeCores > 0, "shared memory requires compute cores")
    require(
      b.memDomain.sharedInputChannels > 0,
      s"sharedInputChannels(${b.memDomain.sharedInputChannels}) must be > 0"
    )
    require(
      b.memDomain.sharedInputChannels % computeCores == 0,
      s"sharedInputChannels(${b.memDomain.sharedInputChannels}) must be divisible by computeCores($computeCores)"
    )
    if (b.memDomain.sharedInputChannels > 32) {
      require(
        b.memDomain.sharedInputChannels == computeCores,
        s"sharedInputChannels(${b.memDomain.sharedInputChannels}) must equal computeCores($computeCores) when > 32"
      )
    }
    val ch = b.memDomain.sharedInputChannels / computeCores
    require(ch > 0, s"channelPerHart($ch) must be > 0")
    ch
  }

  def totalChannel(b: GlobalConfig): Int = if (b.memDomain.sharedEnable) b.memDomain.sharedInputChannels else 0
}
