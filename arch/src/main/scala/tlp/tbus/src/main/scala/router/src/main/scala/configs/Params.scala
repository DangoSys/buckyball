package framework.tlp.tbus.router.configs

import framework.generic.contract.memdomain.MvoverISA

import framework.tlp.tbus.configs.{Params => TBusParam}

import chisel3.util.log2Ceil
import framework.tlp.tbus.configs.Layout
import framework.tlp.shm.{MeshSharedMemParams, MeshThreadAttachment}
import upickle.default.{macroRW, ReadWriter}
import framework.generic.acc.frontend.configs.FrontendParam
import framework.generic.acc.memdomain.configs.MemDomainParam

case class Params(physicalPorts: Int) {

  def mesh(frontendParam: FrontendParam, memParam: MemDomainParam, tbusParam: TBusParam): MeshSharedMemParams = {
    require(
      memParam.bankWidth == 128 && memParam.bankMaskLen == 16
    )
    val totalBanks            = Layout.totalBank(memParam)
    val nThreads              = tbusParam.slots
    val tagBits               = math.max(1, log2Ceil(frontendParam.rob_entries))
    val externalPhysicalPorts = physicalPorts
    val physicalBanks         = totalBanks
    val rows                  = math.ceil(math.sqrt(physicalBanks.toDouble)).toInt
    val columns               = (physicalBanks + rows - 1) / rows
    val perThread             = Layout.portsPerThread(memParam, tbusParam)
    MeshSharedMemParams(
      rows = rows,
      cols = columns,
      entriesPerBank = memParam.sharedBankEntries,
      dataBits = memParam.bankWidth,
      tagBits = tagBits,
      localBankBits = MvoverISA.BankBits,
      threads = (0 until nThreads).map { thread =>
        val ports = if (tbusParam.participants.contains(thread)) perThread else 0
        MeshThreadAttachment(tbusParam.bankNodes(thread), ports)
      },
      externalChannels = externalPhysicalPorts,
      visibleBanks = totalBanks
    )
  }

}

object Params { implicit val rw: ReadWriter[Params] = macroRW }
