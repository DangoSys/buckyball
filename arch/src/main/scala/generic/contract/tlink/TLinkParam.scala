package framework.generic.contract.tlink

import framework.generic.bus.axi4
import framework.generic.memory.spm

/**
 * What a hart knows of a thread: the widths of TLink and the stores the thread reports
 * to the software that manages it. Nothing else of the thread reaches its hart.
 */
case class TLinkParam(
  taskBits:  Int,
  codeBytes: Int,
  dataBase:  BigInt,
  dataBytes: Int,
  tss:       spm.Params,
  dma:       axi4.Params)
