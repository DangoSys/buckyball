package framework.hart.memory.access.configs

import framework.generic.contract.memory.access.AccessParams

import framework.generic.contract.memory.access.mmu.configs.{Params => WalkerParam}

import framework.generic.contract.memory.access.PhysicalRegion

import upickle.default.{macroRW, ReadWriter}

case class Params(
  memory:             AccessParams,
  regions:            Seq[PhysicalRegion],
  translationEntries: Int,
  lineBits:           Int) {
  val walker = WalkerParam(memory.chi, translationEntries)
}

object Params {
  implicit val chiRW:    ReadWriter[framework.generic.bus.chi.Params] = macroRW
  implicit val regionRW: ReadWriter[PhysicalRegion]                   = macroRW
  implicit val memoryRW: ReadWriter[AccessParams]                     = macroRW
  implicit val rw:       ReadWriter[Params]                           = macroRW
}
