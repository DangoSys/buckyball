package framework.memdomain.boot

import framework.memdomain.isa.MemISA
import framework.system.core.rocket.BuckyballCommand
import framework.top.GlobalConfig

/** Internal MemDomain boot command program for one accelerator instance. */
object MemBoot {
  private val BootBankId = 0

  def initializationCommands(b: GlobalConfig): Seq[BuckyballCommand] =
    Seq(MemISA.mset(MemISA.MsetArgs(BootBankId, columns = b.memDomain.bankNum, alloc = true, clear = true)))

  def releaseCommands(b: GlobalConfig): Seq[BuckyballCommand] =
    Seq(MemISA.mset(MemISA.MsetArgs(BootBankId, columns = 0, alloc = false, clear = false)))
}
