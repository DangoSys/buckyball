package framework.core.tbus.configs

import framework.acc.memdomain.configs.MemDomainParam
import framework.core.tbus.configs.{Params => TBusParam}

import upickle.default.{macroRW, ReadWriter}

case class Params(
  router:       framework.core.tbus.router.configs.Params,
  participants: Seq[Int],
  slots:        Int,
  bankNodes:    Seq[Int],
  externalMove: Boolean)

object Params { implicit val rw: ReadWriter[Params] = macroRW }

object Layout {

  def totalBank(memParam: MemDomainParam): Int = {
    require(memParam.sharedEnable)
    require(memParam.sharedBankNum > 0)
    memParam.sharedBankNum
  }

  def portsPerThread(memParam: MemDomainParam, tbusParam: TBusParam): Int = {
    if (!memParam.sharedEnable) return 0
    val threads = tbusParam.participants.size
    require(threads > 0)
    require(memParam.sharedInputChannels > 0)
    require(memParam.sharedInputChannels % threads == 0)
    val channels = memParam.sharedInputChannels / threads
    require(channels > 0)
    channels
  }

  def totalChannel(memParam: MemDomainParam): Int =
    if (memParam.sharedEnable)
      memParam.sharedInputChannels
    else 0

}
